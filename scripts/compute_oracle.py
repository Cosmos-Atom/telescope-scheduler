"""
Reproduces the greedy-oracle denominator values used in the medium and hard graders,
and reports the practical maximum score for the easy task.

Run from the repo root:
    python scripts/compute_oracle.py

Expected output:
    Easy oracle  (44 steps, seed=42): 19 unique planets observed (max score 0.950)
      Note: HD 108874 b never rises above 30° — max achievable n_observed = 19/20
    Medium oracle (32 steps, seed=7):  182.0
    Hard oracle   (18 steps, seed=13): 133.0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.core import _TelescopeCore, load_planet_dataframe, altitude_from_ra_dec
from server.environment import TASK_CONFIGS


def greedy_oracle(task_id: str) -> float:
    """Run a greedy policy (highest-priority visible unobserved target each step)
    and return the total_priority_observed at episode end."""
    df = load_planet_dataframe()
    cfg = TASK_CONFIGS[task_id]
    core = _TelescopeCore(df, cfg)
    core.reset(seed=cfg["default_seed"])

    for _ in range(cfg["max_steps"]):
        best_idx = -1
        best_priority = -1.0
        for idx, row in core.planets.iterrows():
            if row.observed_tonight:
                continue
            alt = altitude_from_ra_dec(row.ra, row.dec, core.current_time, core.location)
            if alt > 30 and core.weather_state != 2:
                if float(row.priority_score) > best_priority:
                    best_priority = float(row.priority_score)
                    best_idx = idx

        action = best_idx if best_idx >= 0 else core.num_planets
        core.step(action)

        if core.done:
            break

    return core.total_priority_observed


def easy_oracle() -> int:
    """Return the number of unique planets observable in the easy task window."""
    df = load_planet_dataframe()
    cfg = TASK_CONFIGS["easy"]
    core = _TelescopeCore(df, cfg)
    core.reset(seed=cfg["default_seed"])

    for _ in range(cfg["max_steps"]):
        best_idx = -1
        best_priority = -1.0
        for idx, row in core.planets.iterrows():
            if row.observed_tonight:
                continue
            alt = altitude_from_ra_dec(row.ra, row.dec, core.current_time, core.location)
            if alt > 30:
                if float(row.priority_score) > best_priority:
                    best_priority = float(row.priority_score)
                    best_idx = idx

        action = best_idx if best_idx >= 0 else core.num_planets
        core.step(action)

        if core.done:
            break

    return core.n_observed_tonight


if __name__ == "__main__":
    easy_n = easy_oracle()
    medium_val = greedy_oracle("medium")
    hard_val = greedy_oracle("hard")
    print(f"Easy oracle  (44 steps, seed=42): {easy_n} unique planets observed (max score {easy_n/20:.3f})")
    print(f"  Note: HD 108874 b never rises above 30° — max achievable n_observed = {easy_n}/20")
    print(f"Medium oracle (32 steps, seed=7):  {medium_val}")
    print(f"Hard oracle   (18 steps, seed=13): {hard_val}")
