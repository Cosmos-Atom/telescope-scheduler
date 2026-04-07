import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from openenv.core.env_server import create_fastapi_app
from environment import TelescopeSchedulingEnvironment, TASK_CONFIGS
from models import ObserveAction, TelescopeObservation, TelescopeState

app = create_fastapi_app(TelescopeSchedulingEnvironment, ObserveAction, TelescopeObservation)


@app.post("/grade")
def grade(state: TelescopeState):
    """Compute the grade for a completed episode given a TelescopeState snapshot.

    Accepts the TelescopeState JSON body returned by GET /state at episode end.
    Scores are bounded to [0.0, 1.0] so external harnesses can verify results
    without running inference.py.
    """
    env = TelescopeSchedulingEnvironment()
    env._state = state
    score = env.compute_grade()
    return {"task_id": state.task_id, "score": score}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
