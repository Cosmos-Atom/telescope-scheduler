"""
OpenEnv Environment: TelescopeSchedulingEnvironment
Wraps _TelescopeCore and exposes the reset()/step()/state API.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import Environment

from models import ObserveAction, PlanetInfo, TelescopeObservation, TelescopeState
from server.core import _TelescopeCore, load_planet_dataframe

WEATHER_NAMES = {0: "clear", 1: "partial", 2: "bad"}

TASK_CONFIGS = {
    "easy": {
        "max_steps": 44,
        "weather_locked": True,
        "start_offset_min": 0,
        "deadline_step_cutoff": None,
        "default_seed": 42,
    },
    "medium": {
        "max_steps": 32,
        "weather_locked": False,
        "start_offset_min": 0,
        "deadline_step_cutoff": None,
        "default_seed": 7,
    },
    "hard": {
        "max_steps": 18,
        "weather_locked": False,
        "start_offset_min": 30,
        "deadline_step_cutoff": 5,
        "default_seed": 13,
    },
}

# Module-level singleton DataFrame — loaded once on first use
_DF = None


def _get_df():
    global _DF
    if _DF is None:
        _DF = load_planet_dataframe()
    return _DF


class TelescopeSchedulingEnvironment(Environment):
    """
    Telescope observation scheduling environment.

    An AI agent must decide which exoplanet targets to observe each night,
    balancing scientific priority, atmospheric conditions, weather, and deadlines.

    Tasks:
      easy   — perfect weather, full night, maximise unique targets observed
      medium — stochastic weather, partial night, maximise priority-weighted score
      hard   — tight window, soft deadlines on top-3 targets, priority + urgency
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_id: str = "easy"
        self._core: _TelescopeCore = None
        self._state = TelescopeState()

    # ------------------------------------------------------------------ #
    # OpenEnv interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed=None,
        episode_id=None,
        task_id: str = "easy",
        **kwargs,
    ) -> TelescopeObservation:
        task_config = TASK_CONFIGS.get(task_id, TASK_CONFIGS["easy"])
        if seed is None:
            seed = task_config["default_seed"]

        self._task_id = task_id
        df = _get_df()
        self._core = _TelescopeCore(df, task_config)
        self._core.reset(int(seed))

        self._state = TelescopeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            weather_state=self._core.weather_state,
            current_time_str=self._core.current_time.strftime("%H:%M"),
            n_observed_tonight=0,
            total_priority_observed=0.0,
            episode_reward=0.0,
            max_steps=task_config["max_steps"],
            deadlines_met_before_cutoff=0,
        )

        return self._build_obs(
            last_action_type="reset",
            last_target_name=None,
            reward=None,
            done=False,
            last_reward_components={},
        )

    def step(
        self,
        action: ObserveAction,
        timeout_s=None,
        **kwargs,
    ) -> TelescopeObservation:
        if self._core is None:
            raise RuntimeError("Call reset() before step().")

        reward, done, info = self._core.step(action.target)

        self._state.step_count = self._core.step_count
        self._state.weather_state = self._core.weather_state
        self._state.current_time_str = self._core.current_time.strftime("%H:%M")
        self._state.n_observed_tonight = self._core.n_observed_tonight
        self._state.total_priority_observed = round(self._core.total_priority_observed, 2)
        self._state.episode_reward = round(self._core.episode_reward, 4)
        self._state.deadlines_met_before_cutoff = self._core.deadlines_met_before_cutoff

        return self._build_obs(
            last_action_type=info.get("action_type", "unknown"),
            last_target_name=info.get("planet_name"),
            reward=round(reward, 4),
            done=done,
            last_reward_components=info.get("reward_components", {}),
        )

    @property
    def state(self) -> TelescopeState:
        return self._state

    def compute_grade(self) -> float:
        """Server-side grader — deterministic, bounded to (0.0, 1.0) exclusive.

        Denominators (oracle upper bounds measured empirically with greedy policy):
          easy   — 20 planets total in dataset
          medium — 182 = max priority sum achievable in 32 steps (greedy, seed=7)
          hard   — deadline_score: 3 soft-deadline targets
                   priority_score: 133 = oracle value for 18 steps greedy-deadline policy
        """
        _EPS = 1e-4
        state = self._state
        if state.task_id == "easy":
            raw = min(state.n_observed_tonight / 20.0, 1.0)
        elif state.task_id == "medium":
            raw = min(state.total_priority_observed / 182.0, 1.0)
        else:  # hard
            deadline_score = min(getattr(state, "deadlines_met_before_cutoff", 0) / 3.0, 1.0)
            priority_score = min(state.total_priority_observed / 133.0, 1.0)
            raw = 0.6 * deadline_score + 0.4 * priority_score
        # Scores must be strictly within (0, 1) per OpenEnv validator requirements
        return round(max(_EPS, min(raw, 1.0 - _EPS)), 4)

    def get_tasks(self) -> list:
        """Return metadata for all supported tasks."""
        return [
            {
                "task_id": "easy",
                "description": "Clear Night Harvest — perfect weather, full 11-hr night, no deadlines. Maximise unique targets observed.",
                "max_steps": TASK_CONFIGS["easy"]["max_steps"],
            },
            {
                "task_id": "medium",
                "description": "Priority Queue Night — stochastic weather, 8-hr partial night. Maximise priority-weighted observation score.",
                "max_steps": TASK_CONFIGS["medium"]["max_steps"],
            },
            {
                "task_id": "hard",
                "description": "Tight Window Deadline Pressure — 4.5-hr window, top-3 targets have soft deadlines. Balance urgency and priority.",
                "max_steps": TASK_CONFIGS["hard"]["max_steps"],
            },
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_obs(
        self,
        last_action_type: str,
        last_target_name,
        reward,
        done: bool,
        last_reward_components: dict,
    ) -> TelescopeObservation:
        core = self._core
        planet_dicts = core.get_planet_infos()
        planet_models = [PlanetInfo(**d) for d in planet_dicts]

        hrs_until_sunrise = (
            core.sunrise - core.current_time
        ).total_seconds() / 3600.0
        weather_str = WEATHER_NAMES[core.weather_state]

        n_deadline_remaining = sum(
            1
            for p in planet_models
            if p.has_deadline
            and p.deadline_status == "before_deadline"
            and not p.observed_tonight
        )

        narrative = _build_narrative(
            current_time_str=core.current_time.strftime("%H:%M"),
            hours_until_sunrise=round(hrs_until_sunrise, 2),
            weather=weather_str,
            planets=planet_models,
            last_action_type=last_action_type,
            last_target_name=last_target_name,
            reward=reward,
            step_count=core.step_count,
            max_steps=core.max_steps,
        )

        return TelescopeObservation(
            done=done,
            reward=reward,
            narrative=narrative,
            current_time_str=core.current_time.strftime("%H:%M"),
            hours_until_sunrise=round(hrs_until_sunrise, 2),
            weather=weather_str,
            weather_state=core.weather_state,
            planets=planet_models,
            last_action_type=last_action_type,
            last_target_name=last_target_name,
            last_reward_components=last_reward_components,
            n_observed_tonight=core.n_observed_tonight,
            total_priority_observed=round(core.total_priority_observed, 2),
            n_deadline_targets_remaining=n_deadline_remaining,
            step_count=core.step_count,
            max_steps=core.max_steps,
        )


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(
    current_time_str: str,
    hours_until_sunrise: float,
    weather: str,
    planets: list,
    last_action_type: str,
    last_target_name,
    reward,
    step_count: int,
    max_steps: int,
) -> str:
    lines = [
        "TELESCOPE SCHEDULER — Mauna Kea Observatory",
        f"Time: {current_time_str} | Weather: {weather} | "
        f"{hours_until_sunrise:.2f} hours until sunrise | Step {step_count}/{max_steps}",
        "",
    ]

    if last_action_type == "reset":
        lines.append("Episode started. Make your first observation decision.")
    elif last_action_type == "observe" and last_target_name:
        reward_str = f"{reward:+.3f}" if reward is not None else "?"
        lines.append(f"Last action: observed {last_target_name}  (reward: {reward_str})")
    elif last_action_type == "wait":
        lines.append("Last action: waited one slot  (penalty: -0.011)")
    elif last_action_type == "invalid":
        lines.append(
            "Last action: INVALID — target was below horizon or weather was bad  (penalty: -0.033)"
        )
    lines.append("")

    # Observable targets
    observable = [p for p in planets if p.visible]
    if observable:
        lines.append("OBSERVABLE NOW (altitude > 30°, weather permits):")
        for i, p in enumerate(planets):
            if not p.visible:
                continue
            idx = planets.index(p)
            flags = []
            if not p.observed_tonight:
                flags.append("NOT YET OBSERVED")
            else:
                flags.append("already observed")
            if p.has_deadline:
                flags.append(f"DEADLINE ({p.deadline_status.upper()})")
            flag_str = " | ".join(flags)
            lines.append(
                f"  [{idx:2d}] {p.name:<20} "
                f"Priority {p.priority_score:>2} | "
                f"Alt {p.altitude_deg:>5.1f}° | "
                f"Airmass {p.airmass:.2f} | "
                f"{flag_str}"
            )
    else:
        lines.append("OBSERVABLE NOW: none (all targets below horizon or weather is bad)")

    lines.append("")

    # Not yet observable
    not_visible = [p for p in planets if not p.visible]
    if not_visible:
        lines.append("NOT YET OBSERVABLE:")
        for p in not_visible:
            idx = planets.index(p)
            if p.time_until_set_hr >= 11.9:
                timing = "rises later"
            else:
                timing = f"sets in {p.time_until_set_hr:.1f}h"
            lines.append(
                f"  [{idx:2d}] {p.name:<20} "
                f"Priority {p.priority_score:>2} | "
                f"Alt {p.altitude_deg:>5.1f}°  ({timing})"
            )
        lines.append("")

    lines.append(
        "ACTION: Reply with the INDEX NUMBER (0–19) to observe that target, "
        "or 20 to WAIT one 15-minute slot."
    )

    return "\n".join(lines)
