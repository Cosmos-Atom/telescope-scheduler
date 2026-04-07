"""
Client for the Telescope Scheduling environment.
"""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import ObserveAction, TelescopeObservation, TelescopeState


class TelescopeEnv(EnvClient[ObserveAction, TelescopeObservation, TelescopeState]):

    def _step_payload(self, action: ObserveAction) -> dict:
        return {"target": action.target, "task_id": action.task_id}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=TelescopeObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> TelescopeState:
        return TelescopeState(**payload)
