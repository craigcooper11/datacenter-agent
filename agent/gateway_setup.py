"""Create the agent's MLflow AI Gateway endpoint, served by the Ollama model to start.

Run: `docker compose exec agent uv run --no-sync python -m agent.gateway_setup`.

After that, manage the endpoint in the MLflow UI (AI Gateway > Endpoints), e.g. make OpenAI the
primary model and keep this Ollama model as the fallback. The agent only knows the endpoint name.

Idempotent: reuses the secret, model definition and endpoint if they already exist. MLflow has
no public client API for the gateway yet, so this uses the tracking REST store, the same API the
MLflow UI calls.
"""

import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities.gateway_endpoint import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.tracking._tracking_service.utils import _get_store

from agent.config import get_settings


def main() -> None:
    s = get_settings()
    if not s.mlflow_tracking_uri or not s.ollama_base_url:
        sys.exit("Needs MLFLOW_TRACKING_URI and OLLAMA_BASE_URL in .env")
    import mlflow

    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    store = _get_store()
    name = s.mlflow_gateway_endpoint

    existing = next((e for e in store.list_gateway_endpoints() if e.name == name), None)
    if existing:
        print(f"Endpoint '{name}' already exists ({existing.endpoint_id}); nothing to do.")
        return

    secret_name = f"{name}-ollama"
    secret = next((x for x in store.list_secret_infos(provider="ollama") if x.secret_name == secret_name), None)
    if secret is None:
        # Ollama needs no key; the provider treats "ollama" as "no Authorization header".
        secret = store.create_gateway_secret(
            secret_name=secret_name,
            secret_value={"api_key": "ollama"},
            provider="ollama",
            auth_config={"api_base": s.ollama_base_url.rstrip("/") + "/v1"},
        )

    model_def_name = f"{name}-{s.ollama_model}"
    model_def = next((m for m in store.list_gateway_model_definitions() if m.name == model_def_name), None)
    if model_def is None:
        model_def = store.create_gateway_model_definition(
            name=model_def_name, secret_id=secret.secret_id, provider="ollama", model_name=s.ollama_model
        )

    endpoint = store.create_gateway_endpoint(
        name=name,
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
            )
        ],
    )
    print(f"Created endpoint '{endpoint.name}' → ollama/{s.ollama_model} at {s.ollama_base_url}")


if __name__ == "__main__":
    main()
