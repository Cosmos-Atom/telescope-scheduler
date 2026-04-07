import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from openenv.core.env_server import create_fastapi_app
from environment import TelescopeSchedulingEnvironment
from models import ObserveAction, TelescopeObservation

app = create_fastapi_app(TelescopeSchedulingEnvironment, ObserveAction, TelescopeObservation)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
