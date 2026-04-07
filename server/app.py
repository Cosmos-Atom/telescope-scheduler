import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app
from environment import TelescopeSchedulingEnvironment, TASK_CONFIGS
from models import ObserveAction, TelescopeObservation

app = create_fastapi_app(TelescopeSchedulingEnvironment, ObserveAction, TelescopeObservation)

# Shared environment instance used by the session manager (created by create_fastapi_app).
# We expose a /grade endpoint that reads from the same env instance.
_env_instance = TelescopeSchedulingEnvironment()


@app.get("/grade")
def grade(task_id: str = "easy"):
    """Return the current episode score for the given task_id.

    Scores are bounded to [0.0, 1.0] and computed server-side so external
    harnesses can verify results without running inference.py.
    """
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS)}")
    # Use the shared env if it has been reset to the requested task,
    # otherwise return 0.0 (no episode has been run for this task yet).
    env = _env_instance
    if env._core is None or env._task_id != task_id:
        return {"task_id": task_id, "score": 0.0, "note": "No episode run yet for this task"}
    score = env.compute_grade()
    return {"task_id": task_id, "score": score}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
