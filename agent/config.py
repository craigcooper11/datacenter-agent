"""Environment-driven settings. Everything the agent talks to is external and configured here."""

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from agent.tools import ActionScope

# Export .env into os.environ so libraries that read it directly (OpenAI, MLflow) see it too.
load_dotenv()


class Settings(BaseSettings):
    # "live" talks to your Proxmox and Grafana; "demo" runs against a simulated homelab (agent/demo.py)
    # with a disk filling up on cue, so anyone can try the agent without infrastructure.
    backend: Literal["live", "demo"] = "live"

    # LLM. "ollama" calls the Ollama server directly; "gateway" goes through the MLflow AI Gateway
    # endpoint below (which fronts Ollama; see agent/gateway_setup.py) so calls get usage tracking.
    llm_provider: Literal["openai", "ollama", "gateway"] = "openai"
    # Named for its role, not a model: which models serve it (and fallbacks) is set in the MLflow UI.
    mlflow_gateway_endpoint: str = "datacenter-agent"
    # "none" lets reasoning models (e.g. gpt-5.x) use tools on Chat Completions, and skips qwen's
    # thinking step. Leave unset for models that reject the parameter (e.g. gpt-4.1).
    llm_reasoning_effort: str | None = None
    openai_model: str = "gpt-4.1"  # OPENAI_API_KEY is read by the OpenAI client itself
    ollama_model: str = "qwen3.5:latest"  # needs a tool-calling model
    # Context window to request (direct Ollama only; for the gateway, set it on the Ollama server).
    ollama_num_ctx: int = 16384
    ollama_base_url: str | None = None  # the Ollama server, e.g. http://ollama.lan:11434

    # Proxmox (proxmoxer, API token auth). Optional here so evals can run without live infra;
    # LiveBackend validates them.
    proxmox_host: str | None = None
    proxmox_user: str | None = None  # e.g. agent@pve
    proxmox_token_name: str | None = None
    proxmox_token_value: str | None = None
    proxmox_verify_ssl: bool = False
    proxmox_node: str | None = None
    # Which containers the agent may change (restart or resize), always after approval: "all", a list
    # like "104,105", or empty for none (diagnose only). Protected IDs can never be changed.
    proxmox_managed_vmids: str = Field(
        default="", validation_alias=AliasChoices("proxmox_managed_vmids", "proxmox_restartable_vmids")
    )
    proxmox_protected_vmids: str = ""

    def action_scope(self) -> "ActionScope":
        from agent.tools import ActionScope

        def ids(raw: str) -> frozenset[int]:
            return frozenset(int(v) for v in raw.replace(" ", "").split(",") if v)

        raw = self.proxmox_managed_vmids.strip().lower()
        return ActionScope(vmids=None if raw == "all" else ids(raw), protected=ids(self.proxmox_protected_vmids))

    # Grafana, via an externally running mcp-grafana (streamable-http)
    grafana_mcp_url: str | None = None  # e.g. http://mcp-grafana.lan:8000/mcp
    # Bearer token mcp-grafana requires from callers (its --server-auth-token).
    grafana_mcp_token: str | None = None
    grafana_datasource_uid: str = "prometheus"
    # PromQL for per-LXC disk usage %, as exposed by prometheus-pve-exporter.
    # Must return one series per guest with an `id` label like "lxc/101".
    disk_usage_promql: str = '100 * pve_disk_usage_bytes{id=~"lxc/.*"} / pve_disk_size_bytes{id=~"lxc/.*"}'
    # VM disk sizes. Proxmox can't see usage inside a VM without the QEMU guest agent, so VMs are
    # listed with size only.
    vm_disk_size_promql: str = 'pve_disk_size_bytes{id=~"qemu/.*"}'
    # Joined on `id` to attach guest names; set empty to skip.
    guest_info_promql: str = "pve_guest_info"

    # Tracing. Unset means MLflow's local default store.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "datacenter-agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def model_label(s: Settings) -> str:
    if s.llm_provider == "gateway":
        return f"mlflow-gateway/{s.mlflow_gateway_endpoint}"
    return f"ollama/{s.ollama_model}" if s.llm_provider == "ollama" else f"openai/{s.openai_model}"
