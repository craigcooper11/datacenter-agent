# AGENTS.md — Datacenter Agent

## Context

A chat agent for a Proxmox homelab. It diagnoses disk-space problems on an LXC container using
real Grafana metrics, confirms them against the Proxmox API, and remediates them by restarting
the container — but only after an explicit human approval gate.

Keep changes focused on reliable end-to-end behavior. Destructive actions must remain explicitly
approval-gated, tool behavior must stay inspectable, and live infrastructure access must retain
the narrowest practical scope.

## The idea, and why this shape

I already run a real Proxmox homelab. The agent:

1. **Diagnoses** a disk-space-exhaustion fault on an LXC container by querying Grafana (via
   Grafana's own open-source `mcp-grafana` MCP server, driven by Prometheus + a Proxmox exporter)
2. **Confirms** via a direct Proxmox API call (not just metrics — actual container state)
3. **Acts**: restarts the affected container — but only after an explicit human-in-the-loop
   confirmation gate implemented with LangGraph's `interrupt()` primitive

### Why this is a focused workflow

A general-purpose root-cause-analysis assistant is difficult to evaluate and easy to make vague.
Tying the agent to a concrete infrastructure fault and a gated remediation action keeps tool
boundaries, approval behavior, and evaluation scenarios measurable.

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Native `interrupt()`/resume support = the human-in-the-loop gate, which is the centerpiece design choice |
| UI | **Chainlit** (`agent/app.py`, in docker compose), plus a **terminal CLI** (`python -m agent.cli`) | Chainlit is the demo surface: tool calls as expandable steps, the interrupt as Approve/Deny buttons. The CLI is a no-browser fallback. Both drive the same `run_turn()`, so the gate lives in the graph, not the UI |
| Observability metrics source | **Grafana**, via **grafana/mcp-grafana** (Grafana's own OSS MCP server), run as a compose service (`grafana/mcp-grafana:1.6.0`, read-only, bearer-token auth) and connected via `langchain-mcp-adapters` (streamable-http) | Keeps the Grafana part of the stack literal — it's their own tool doing the work, not just their name in a diagram |
| Metrics pipeline feeding Grafana | **Prometheus** + **prometheus-pve-exporter** (community Proxmox exporter), scraping the real Proxmox host — **runs outside this repo** | Real data, no synthetic/mocked metrics |
| Proxmox actions (status check + restart) | **`proxmoxer`** Python library, called directly from a custom tool — not via MCP | Simpler and more reliable than standing up a second MCP server for one write action; a good trade-off to discuss (why direct API here but MCP for Grafana) |
| LLM provider | **OpenAI** by default; the Ollama server can be used either directly (`LLM_PROVIDER=ollama`) or through the **MLflow AI Gateway** (`LLM_PROVIDER=gateway`, endpoint created by `python -m agent.gateway_setup`) | Provides a capable default while preserving a local-model path. The gateway adds usage tracking for every LLM call. Ollama runs on a separate server, so there is no localhost default |
| Tracing | **MLflow** server in docker compose (host port 5001), or any server at `MLFLOW_TRACKING_URI`, with `mlflow.langchain.autolog()` in the agent | Provides auditable traces for reviewing the behavior of an agent that can take real actions |
| Evaluation | **`mlflow.genai.evaluate`** over a scenario suite, run against a scripted fixture backend, with code scorers (safety/trajectory) + LLM-judge `Guidelines` scorers | Repeatable and safe: evals must never restart real containers. Code scorers check the hard invariants (gate respected, no restart when healthy/denied/out-of-scope); judges check answer quality |
| Fault scenario | **Disk space exhaustion** on one dedicated sandbox LXC container (not a production VM) | Safe to trigger live; clear, demoable metric (disk %) |
| Act scope | **Restart the container only.** Do NOT build VM creation/recreation logic. | Keeps the destructive surface small and the action path reliable. Replacing a guest when restart does not help remains out of scope |
| Repo scope | **The agent, plus a compose file that runs it with its own supporting services** (mcp-grafana, MLflow). Grafana, Prometheus and pve-exporter are external homelab infrastructure, reached via URLs in `.env` | Keeps the repo focused on the agent itself, not my homelab's infrastructure |
| Packaging | **docker compose** (`agent` Chainlit service + `mlflow` + `mcp-grafana`) for running; **`uv` + `pyproject.toml`** for local dev, the CLI and evals | Easy to stand up: fill `.env`, `docker compose up -d --build` |

## Repo layout

```
datacenter-agent/              # the agent itself is named "Datacenter Agent"
├── AGENTS.md                  # this file
├── README.md                  # setup + demo script
├── pyproject.toml
├── Dockerfile                 # agent image (uv + Chainlit)
├── docker-compose.yml         # agent (Chainlit, :8000) + mcp-grafana (:8001) + mlflow (:5001)
├── chainlit.md                # Chainlit readme panel
├── .env.example
├── .env                       # gitignored — real secrets
├── .gitignore
├── agent/
│   ├── config.py              # env var loading
│   ├── backends.py            # LiveBackend: mcp-grafana client + proxmoxer
│   ├── demo.py                # simulated homelab backend (demo mode, and evals)
│   ├── tools.py                # query_disk_usage, get_container_status, restart_container (interrupt() gate)
│   ├── graph.py                # LangGraph StateGraph: agent node + tool node, MemorySaver, MLflow tracing
│   ├── app.py                  # Chainlit UI: tool calls as steps, Approve/Deny on interrupt
│   ├── gateway_setup.py        # creates the MLflow AI Gateway endpoint fronting Ollama (idempotent)
│   └── cli.py                  # terminal REPL: streams tool calls, [y/N] on interrupt
└── evals/
    ├── scenarios.py            # scenario dataset (healthy, disk-full, denied, out-of-scope, ...)
    ├── scorers.py               # code scorers + LLM-judge Guidelines
    └── run_evals.py             # mlflow.genai.evaluate entrypoint
```

## Key implementation notes

**LangGraph interrupt pattern** — do NOT use `interrupt_before=["tools"]` on the whole graph (that
pauses before *every* tool call, including read-only ones). Instead, call
`langgraph.types.interrupt()` **inside** the `restart_container` tool function itself, so
Grafana/Proxmox read-only queries run freely and only the destructive action pauses. Use
`MemorySaver` as the checkpointer so interrupt/resume works across the pause.

**UI + interrupt** — `graph.run_turn()` streams the graph and, when it yields `__interrupt__`,
awaits a `decide` callback, then resumes with `Command(resume={"approved": bool})`. Chainlit's
`decide` shows `cl.AskActionMessage` Approve/Deny buttons (a timeout counts as deny); the CLI's
prompts `[y/N]`; evals use a scripted policy. Tools take a backend object so the same graph runs
against live infra or a simulated one.

**One tool call per step** — on resume, every tool call in the interrupted step re-runs. OpenAI
gets `parallel_tool_calls=False`; for other providers the agent node trims to the first tool call.

**mcp-grafana connection** — `grafana/mcp-grafana` runs in compose in `--transport
streamable-http` mode with `--disable-write`, `--enabled-tools=prometheus,datasource`, and
`--allowed-hosts` for both the agent container and host-side CLI/evals. It gets only `GRAFANA_URL`
and a Viewer `GRAFANA_SERVICE_ACCOUNT_TOKEN`, not the whole `.env`. Callers must send
`GRAFANA_MCP_TOKEN` as a bearer token; the agent connects via
`langchain_mcp_adapters.client.MultiServerMCPClient`.

**proxmoxer auth** — use a scoped API token, not the root user: read access everywhere
(`PVEAuditor` on `/`), `VM.PowerMgmt` only on the containers in `PROXMOX_MANAGED_VMIDS`. This
keeps the token's blast radius small even before the code-level scope check or the approval gate
come into play.

## Env vars

See `.env.example` for the full annotated list: `LLM_PROVIDER`, `MLFLOW_GATEWAY_ENDPOINT`,
`MLFLOW_CRYPTO_KEK_PASSPHRASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `PROXMOX_HOST`, `PROXMOX_USER`, `PROXMOX_TOKEN_NAME`, `PROXMOX_TOKEN_VALUE`,
`PROXMOX_VERIFY_SSL`, `PROXMOX_NODE`, `PROXMOX_MANAGED_VMIDS`, `PROXMOX_PROTECTED_VMIDS`,
`GRAFANA_URL`, `GRAFANA_SERVICE_ACCOUNT_TOKEN`, `GRAFANA_MCP_URL`, `GRAFANA_MCP_TOKEN`,
`GRAFANA_DATASOURCE_UID`, `MLFLOW_TRACKING_URI`, `EVAL_JUDGE_MODEL`. Compose overrides
`GRAFANA_MCP_URL` and `MLFLOW_TRACKING_URI` for the agent container; the `.env` values are for
host-side runs.

## Development commands

```bash
uv sync --frozen
uv run ruff check agent evals tests
uv run ruff format --check agent evals tests
uv run pytest -q
docker compose up -d --build
```

## Repository constraints

- Don't build multi-agent orchestration (supervisor/swarm) — a single ReAct-style agent is the
  intended scope
- Don't add VM creation/provisioning as an act option — restart only
- Don't synthesize fake data in the agent or the live demo — use the real homelab. The one
  exception is the simulated backend, which exists so evals and the no-homelab demo mode are
  repeatable and never touch real containers
- Don't add monitoring/infra config (Prometheus, Grafana provisioning, dashboards) to this repo —
  only the agent and its own services (mcp-grafana, MLflow)
- Don't run or call Ollama on this Mac — it runs on my server; never add localhost or
  host.docker.internal defaults for it
