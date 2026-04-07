"""
Reproduces the greedy-oracle denominator values used in the medium and hard graders.

Run from the repo root:
    python scripts/compute_oracle.py

Expected output:
    Medium oracle (32 steps, seed=7):  182.0
    Hard oracle   (18 steps, seed=13): 133.0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.core import _TelescopeCore, load_planet_dataframe
from server.environment import TASK_CONFIGS


def greedy_oracle(task_id: str) -> float:
    """Run a greedy policy (highest-priority visible unobserved target each step)
    and return the total_priority_observed at episode end."""
    df = load_planet_dataframe()
    cfg = TASK_CONFIGS[task_id]
    core = _TelescopeCore(df, cfg)
    core.reset(seed=cfg["default_seed"])

    for _ in range(cfg["max_steps"]):
        # Build candidate list: visible, unobserved, highest priority first
        best_idx = -1
        best_priority = -1.0
        for idx, row in core.planets.iterrows():
            if row.observed_tonight:
                continue
            from server.core import altitude_from_ra_dec
            alt = altitude_from_ra_dec(row.ra, row.dec, core.current_time, core.location)
            if alt > 30 and core.weather_state != 2:
                if float(row.priority_score) > best_priority:
                    best_priority = float(row.priority_score)
                    best_idx = idx

        # Fall back to wait if nothing is visible/observable
        action = best_idx if best_idx >= 0 else core.num_planets
        core.step(action)

        if core.done:
            break

    return core.total_priority_observed


if __name__ == "__main__":
    medium_val = greedy_oracle("medium")
    hard_val = greedy_oracle("hard")
    print(f"Medium oracle (32 steps, seed=7):  {medium_val}")
    print(f"Hard oracle   (18 steps, seed=13): {hard_val}")
