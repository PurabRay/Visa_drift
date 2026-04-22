"""
FastAPI application for the VisaDrift environment (FIXED).

Key fix: max_concurrent_envs lowered from 100 to 1 to match
SUPPORTS_CONCURRENT_SESSIONS=False on the environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: pip install openenv-core") from e

try:
    from ..models import VisaAction, VisaObservation
    from ..environment import VisaEnvironment
except (ImportError, ModuleNotFoundError):
    from models import VisaAction, VisaObservation
    from environment import VisaEnvironment

app = create_app(
    VisaEnvironment,
    VisaAction,
    VisaObservation,
    env_name="visa_drift",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
