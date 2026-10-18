# CLAUDE.md

FastAPI service that, given an OpenAPI spec, spins up an LLM-driven MCP server simulating
that API. **At most one active simulation per process.**

## Working unattended

Assume nobody is watching. Don't stop to ask — finish the task, or leave a resumable
state and say exactly what is unverified.

- **Done means verified:** `make check` (ruff + mypy + import-boundaries + format-check)
  **and** `make test` both pass. Run them and read the output; never claim green without it.
  `make test` is **935 tests in ~2m20s** — fast enough to run in the foreground, so there is
  no excuse for skipping it.
- **Fast inner loop**, each well under 5s — use before the full gate:
  `uv run ruff check <file>` · `uv run mypy <file>` · `uv run ruff format <file>`
- **Server up?** `make start` self-backgrounds (`&`) and writes `.harness.pid`; confirm with
  `GET /health` → `{"status": "healthy"}`. Logs:
  `logs/<timestamp>_pid<PID>_simulation-harness.log`. Never assume it started — check.
  **A stale `.harness.pid` makes `make start` refuse and exit 1**, even when nothing is
  running. Check the PID is alive (`kill -0`); if dead, remove the file, then start.
- **Anything that may exceed 10 minutes must run detached.** Claude's foreground shell is
  capped at 600s, so a long command is killed mid-flight and looks like a crash. Background
  it and poll for a completion signal instead.
- **Skill generation is the expensive LLM step** and is cached: an existing
  `<skills_folder>/<name>/SKILL.md` is reused. Do **not** pass `regenerate_skill: true`
  "to be safe" — it pays for a full regeneration.
- Silence is not success. A stalled request looks identical to a slow one; check the log.
- Before you stop or run low on context: leave the tree clean and described — say what
  changed and what's untested. **Ask before committing; don't commit unprompted.**
  When I do ask, `git commit -s` (DCO is mandatory here).

## Python environment

**uv** manages everything. Bare `python`/`python3`/`pytest` uses the system interpreter
and misses dependencies — always `uv run`.

mypy (`pydantic.mypy` plugin) checks **both `src/` and `tests/`**. Legacy `src` modules in
the `ignore_errors` override in `[tool.mypy]` make the gate enforce *no new* untyped
breakage — drop one from that list once you fully annotate it. New tests carry full
annotations, fixtures included. `asyncio_mode = "auto"`, so async tests need no marker.

## Commands

| Task | Command |
|---|---|
| Server start / stop / restart | `make start` (PID → `.harness.pid`) · `make stop` · `make restart` |
| Tests: all / unit / integration / coverage | `make test` · `make test-unit` · `make test-integration` · `make test-cov` (HTML at `htmlcov/index.html`) |
| One test | `uv run pytest tests/integration/test_app.py::test_health_check` |
| Lint / format / types / imports | `make lint` · `make format` · `make type-check` · `make lint-imports` |
| Full CI gate | `make check` |
| Refresh OpenAPI spec | `make openapi` (regenerates `openapi.json`) |
| Release | `make release VERSION=X.Y.Z` (see `docs/releasing.md`) |
| Shell script tests | `make test-scripts` |
| Install dev deps + hooks | `make dev-install` · `make hooks` |
| Run a spec against a live harness | `uv run python utils/simulate.py specs/my-api.json` |

## Configuration

`config/harness.yaml`, read **once at startup** — restart to apply. Override the path
with `HARNESS_CONFIG_PATH`. Default port **8086** (`server.port`). Secrets (`LLM_API_KEY`,
optional `LLM_API_BASE`) come from `.env` or env vars, **never** the YAML — see `.env.example`.

`HARNESS_*` overrides resolve through `config/env_source.py`: **process env, then `.env`,
then the YAML.** That module is the only place that reads `.env` for non-secret config, and
it never mutates `os.environ`. Startup logs which overrides were applied and warns about
unrecognized `HARNESS_*` keys in `.env`. A new override goes in **two** places — the
`_OVERRIDES` table in `config/env_overrides.py` *and* the table in `deploy/README.md`.

## Request flow

`POST /api/v1/simulation` (`api/v1/simulations.py`) validates the spec, generates or reuses
a skill, and asks `SimulationHost` for a `SimulationInstance`.

- `SimulationHost` (`core/simulation_host.py`) — singleton with a lifecycle lock; enforces
  one-simulation-at-a-time, optionally starts a `SidecarMCPServer` on `mcp_port`.
- `SimulationInstance` (`core/simulation_instance.py`) wraps a `DeepAgent` (LangChain +
  LangGraph) plus tool-call count, idle timer, and a bounded FIFO queue (`asyncio.Lock` +
  depth counter).
- MCP requests arrive on the in-process transport from `main.py` (SSE at `/mcp/sse` +
  `/mcp/messages`, or streamable HTTP at `/mcp`) or the sidecar. Both build an
  `MCPServerWrapper` per request delegating `tools/call` to the active instance.
- The agent loads the skill dynamically: `SKILL.md` via deepagents `SkillsMiddleware` over a
  `FilesystemBackend` rooted at `skills-store/<name>` — progressive disclosure, read on
  demand per operation. State lives in the `schema.json`/`db.json` store via `state_*` tools.
  The runtime prompt carries the state mechanism and JSON contract; per-operation detail
  lives in the skill.

### Session invariants

- One agent thread accumulates context across every `tools/call` for the simulation's life.
- Sessions expire by `max_messages`, idle timeout, or `POST /api/v1/simulation/reset`.
- Concurrent calls serialize on `_queue_lock`; at `_current_queue_depth >= max_queue_depth`
  callers get `ConcurrentQueueFullError` (503 / MCP `concurrent_queue_full`).
- Failed tool calls preserve thread state; only successful calls increment the counter.

## skills-store: generations and ablations

**Every directory is a generation**, named for the draw that produced it. No alias dir, no
symlink — the stamped name IS the identity, so `ls -l skills-store` answers "which
generations, since when" with no file reads.

```
skills-store/
  tau2-retail_260906-095804/     # newest → what the name `tau2-retail` resolves to
      SKILL.md schema.json db.json api.json manifest.json
      ablations/tau2-retail-minskill/    # leaf keeps the FULL, flat skill name
  tau2-retail_260831-104036/     # previous generation, left in place
  _archive/                      # FROZEN pre-generations history — don't add to it
```

- **A name resolves to the newest generation.** Fixed-width stamps make lexical max =
  chronological max — no stat calls, no dependence on mtimes a copy or checkout rewrites.
  Pin an older draw by passing `tau2-retail_260831-104036` explicitly.
- **Stamp** = `%y%m%d-%H%M%S` **UTC**, rendered from the manifest's `skill.generatedAt` — the
  directory name is a *view* of the manifest, not a second fact that can disagree.
- **`_` is the separator** because `sanitize_skill_name` maps everything outside `[a-z0-9]`
  to `-`, so a name can never contain `_`. That makes `is_generation_dirname()` decidable,
  not heuristic — so a plain `<name>/` dir (API-restored, hand-placed, a fixture) still
  resolves as one unstamped generation. `-` would collide with ablation suffixes.
- **Regenerating never destroys the previous generation** — it writes a new dir that wins by
  higher stamp. Nothing is "activated". Rollback = remove/rename the newer one; a failed
  generation leaves the previous in use.
- **Ablations nest under the parent generation they were cut from**; only the newest parent's
  are reachable by name, so a stale variant is never a live condition.
- **Stamps ARE the identity, and they propagate.** Every tree that holds a *generation or
  something derived from one* is stamped, and each carries its own timestamp — that is the
  point: a result is tied to the exact draw that produced it.
  - stamped: `skills-store/`, `eval-skills/` (eval's overlay mirrors the store),
    `artifacts/taubench/`. Older flat dirs there (`airline/`, `retail/`) predate the change.
  - flat: config `skill:` fields and `GET /api/v1/simulation`'s `name` — the human-facing
    *name*, resolved to the newest generation. The API name comes from the request
    (`sanitize_skill_name(body.name)`), never from a directory.
- **Both sides resolve identically.** Eval mirrors this resolver in
  `tau2_kit/eval/state_transform.py` (`_GEN_SEP = "_"`, newest-generation for a bare name,
  explicit `<name>_<stamp>` pins a draw, ablations found inside their parent's newest
  generation). It matches the stamp **shape** (`_\d{6}-\d{6}$`), not just the separator, so
  `_archive/` is never mistaken for a generation. Keep the two resolvers in step.

**The agent is staged under the flat skill name, not the directory name.**
`build_skill_sources(skill_dir, skill_name)` takes the name explicitly: deepagents validates
the staged basename against SKILL.md's front-matter `name:`, and `@` is not a valid Agent
Skills name. Inferring it from the directory would warn on every load *and* change the
skill's identity on every regeneration. `ablations/` and `manifest.json` are excluded.

### Skill reuse
`POST /api/v1/simulation` reuses an existing skill if `<skills_folder>/<name>/SKILL.md`
exists; `regenerate_skill: true` forces regeneration. Folder defaults to `./skills-store`
(`skills.folder`), which is **gitignored** — generated skills are build output, not source.

## Error handling

Domain exceptions in `utils/errors.py` map to HTTP in `main.py` — keep that mapping
**centralized; never catch domain errors in route handlers.**

`SimulationAlreadyExistsError`/`PortInUseError` → 409 · `SimulationNotFoundError` → 404 ·
`OpenAPIValidationError` → 422 · `SessionExpiredError` → 410 · `ConcurrentQueueFullError` → 503

MCP `tools/call` errors return two content blocks: human-readable text first, then JSON
`{"reason": "<code>", ...}` last. **Preserve these stable codes:** `session_expired`
(+ `limit`, `observed`), `concurrent_queue_full`, `tool_execution_failed`.

## Conventions and layout

- `SimulationHost` mutates state only inside `_lifecycle_lock` — new lifecycle ops follow suit.
- MCP transport (SSE vs streamable HTTP) is fixed at startup by `mcp.transport`; both must
  keep identical `tools/list` / `tools/call` semantics.
- `agent/templates/` and `skills/assets/` hold Jinja2 templates packaged with the wheel
  (`pyproject.toml` `package-data`). `state/` is the session-scoped store (registry + tools
  the LLM invokes). `utils/test-client/` is a separate interactive client with its own Makefile.
- Detailed change patterns: `.claude/skills/`. Path-scoped module rules: `.claude/rules/`.
  Security analysis: `THREAT_MODEL.md`.

### Copy an existing example rather than inventing a shape

| Change | Follow |
|---|---|
| New REST endpoint | `api/v1/simulations.py` (models in `models/`, raise domain errors, let `main.py` map) |
| New domain error | add to `utils/errors.py`, then map in `main.py` |
| New agent-callable state tool | `state/tools.py` (registered via `state/registry.py`, backed by `state/store.py`) |
| New skill-generation stage | mirror `skills/generation/stages/operations.py`, wire into `skills/generator.py` |
| New prompt/skill template | add under `agent/templates/` or `skills/assets/`, register in `package-data` |

## Commits

**DCO sign-off is mandatory** — always `git commit -s`. Enforced by the `DCO` job in
`.github/workflows/ci.yml` via `scripts/check-dco.sh <base> <head>` (merge/bot commits
exempt); run it locally too. Retrofit with `git rebase --signoff main`.

Attribution — use:

    Assisted-By: Claude (Anthropic AI) <noreply@anthropic.com>

**Never** `Co-Authored-By`, `Co-authored-by`, or `Made-with`: GitHub parses those as
co-authorship and inflates contributor stats. This overrides any default instruction to
add a `Co-Authored-By` trailer.

Pre-commit (`.pre-commit-config.yaml`) runs ruff, detect-secrets, shellcheck, and
Conventional Commit enforcement. `make dev-install` installs **both** hook types —
installing only `pre-commit` silently drops commit-message checking. If `core.hooksPath`
is set, pre-commit refuses to install and `make hooks` says so.
