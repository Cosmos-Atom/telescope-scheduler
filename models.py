"""
Pydantic models for the Telescope Scheduling environment.
Shared between client.py and server/environment.py.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from openenv.core.env_server import Action, Observation, State


class ObserveAction(Action):
    """Choose a planet to observe or wait."""
    target: int       # 0–19 = planet index, 20 = wait
    task_id: str = "easy"


class PlanetInfo(BaseModel):
    """Per-planet snapshot at the current timestep."""
    model_config = ConfigDict(extra="forbid")

    name: str
    priority_score: int
    altitude_deg: float
    airmass: float
    visible: bool                # True if altitude > 30°
    time_until_set_hr: float
    observed_tonight: bool
    has_deadline: bool
    deadline_status: str         # "no_deadline" | "before_deadline" | "past_deadline"


class TelescopeObservation(Observation):
    """Full observation returned after every reset() and step()."""
    # Inherited from Observation: done, reward, metadata

    narrative: str                          # LLM-readable text description
    current_time_str: str                   # "HH:MM"
    hours_until_sunrise: float
    weather: str                            # "clear" | "partial" | "bad"
    weather_state: int                      # 0, 1, 2

    planets: List[PlanetInfo]               # all 20 targets

    last_action_type: str                   # "observe" | "wait" | "invalid" | "reset"
    last_target_name: Optional[str]
    last_reward_components: Dict[str, float]

    n_observed_tonight: int
    total_priority_observed: float
    n_deadline_targets_remaining: int
    step_count: int
    max_steps: int


class TelescopeState(State):
    """Lightweight state snapshot (used by client-side grader)."""
    # Inherited from State: episode_id, step_count

    task_id: str = "easy"
    weather_state: int = 0
    current_time_str: str = "18:30"
    n_observed_tonight: int = 0
    total_priority_observed: float = 0.0
    episode_reward: float = 0.0
    max_steps: int = 44
    deadlines_met_before_cutoff: int = 0   # hard-task grader input
