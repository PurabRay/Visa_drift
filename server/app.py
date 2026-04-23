
"""
FastAPI application for the VisaDrift environment.

Exposes VisaEnvironment over HTTP and WebSocket endpoints compatible
with openenv validate and the EnvClient.

Endpoints (auto-created by openenv):
    POST /reset   – reset the environment (pass task_id in kwargs body)
    POST /step    – execute an action
    GET  /state   – current environment state
    GET  /schema  – action / observation JSON schemas
    GET  /health  – health check
    WS   /ws      – WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

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
    max_concurrent_envs=100,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point: uvicorn server.app:app or uv run server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
