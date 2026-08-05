# AI-in-the-Loop Discovery Platform for `habitat_pipeline` — Design Doc

*A merged design. The **core** is a repo-native discovery loop plus a scientific-rigor layer; the **hardening layer** is a right-sized subset of standard MLOps practice. Corrected for the actual repo: multi-animal **electrophysiology** (Kilosort 4 spikes) + video tracking + scored behavior, not calcium imaging.*

---

## 1. Purpose

Turn `habitat_pipeline` into a platform where an AI agent can, with a scientist in the loop, run analyses on a recording session, interpret the results (numbers **and** figures), search the literature, propose and rank hypotheses, implement them as new modules following the repo's existing conventions, re-run on data, and iterate — with every step logged and reviewable.

This document merges two independent designs (one repo-grounded, one ops-focused) into a single plan and drops the parts that are over-engineered for a single lab.

## 2. Design principles

1. **Keep the human at the composition seams.** The best current evidence (Horstmann et al., 2606.07718, a case study on a *neuroscience* data-to-discovery pipeline) is that agents reliably complete individual pipeline *stages* but fail at composing them into correct end-to-end discovery. So the loop automates *within* stages and requires a scientist decision *between* them: which hypothesis to pursue, whether generated code is sound, whether a result is real.

2. **Use what the repo already provides.** `habitat_pipeline` is already an unusually good agent substrate — a tiny data plane (three dataclasses + one sync object), per-module CLIs, a unified result-dict schema, a `(session, params)`-keyed cache, a `database/` layer, and a mock-data test suite. The platform *wires these together*; it does not rebuild them.

3. **Rigor is the differentiator, not agent cleverness.** An agent that generates 50 hypotheses and tests them all against one dataset manufactures false positives fast. Held-out data, pre-registration, and multiple-comparison control are first-class features, not afterthoughts.

4. **Right-size the infrastructure.** Containers, CI, and experiment tracking earn their place. Kubernetes, Terraform, a managed enterprise platform, and RLHF do not — not for one lab on a Janelia workstation + HPC. Add them only if the lab outgrows that.

## 3. Landscape (why this design, briefly)

Direct analogs for the *discovery loop* (not generic MLOps):

- **FutureHouse / Edison "Robin"** — the closest template: literature agents (PaperQA2-based) + a data-analysis agent in a loop that generated a real preclinical discovery. Open source.
- **Google DeepMind Co-Scientist** (Nature, 2026) — the *generate → debate → rank → evolve* multi-agent pattern; the lesson is to use a small society of agents with an explicit critique/ranking step rather than one agent's first idea.
- **Sakana AI Scientist-v2** — the fully-autonomous extreme; useful as a warning about where *not* to remove the human.

MLOps platforms (Domino, W&B, NeuroCAAS, Jupyter AI) are complementary — they inform the hardening layer in §7, not the loop itself.

## 4. Architecture: the discovery loop

A small multi-agent system. Each role maps to concrete repo affordances.

```
        ┌──────────────────────────────────────────────────────────┐
        │                    SCIENTIST (gates ↓)                     │
        └──────────────────────────────────────────────────────────┘
   ①Run ─▶ ②Interpret ─▶ ③Literature ─▶ ④Hypothesize ─▶ ⑤Implement ─▶ ⑥Re-run
     ▲                                        (generate/critique/rank)      │
     └──────────────────────────── loop ─────────────────────────────────┘
                       every iteration written to the lab notebook (§6)
```

| # | Agent role | What it does | Repo hook |
|---|---|---|---|
| ① | **Runner** | Invokes analyses honoring the cache | `python -m ephys.decode_opponent_identity …`, `run_inter_brain`, `run_social_spatial`, `run_partner_distance`; `.gui_cache/` keyed on `(session, params)` |
| ② | **Interpreter** | Reads the result-dict *and looks at the PNGs* (vision) | Unified result-dict schema (`parameters`, `class_label`, `analysis_title`, `behavioral_summary`) from `ephys/_lda_decoding.py`; plots from `ephys/decoding_plots.py` |
| ③ | **Literature** | Searches papers/preprints/targets | MCP tools already connected in this environment: bioRxiv/medRxiv, PubMed-style article search, Open Targets, ChEMBL |
| ④ | **Hypothesizer** | Generate → critique → rank candidate hypotheses | Co-Scientist-style; 2–3 agents debating, scored on a rubric (§5) |
| ⑤ | **Coder** | Writes a new `ephys/` module **+ matching mock test**, following the result-dict contract | Pattern in `decode_*` wrappers; tests in `tests/` (e.g. `test_social_spatial_fields.py`) |
| ⑥ | **Runner** (again) | Re-runs and compares against prior iterations | Same as ① |

**Interpreter caveat.** The neuroscience case study flags *weak visual self-evaluation* as a top failure mode. Mitigation: every figure the pipeline emits must be accompanied by the structured metrics behind it (already true of the result-dict), and the Interpreter reasons over the numbers primarily and the image secondarily — never the image alone.

**Hypothesizer design.** Do not accept an agent's first idea. Run a bounded generate/critique/rank cycle (the Co-Scientist pattern), producing a ranked shortlist with explicit rationale and citations, which the scientist picks from at the gate.

## 5. Scientific-rigor & hypothesis layer (the core differentiator)

This is what separates a research tool from a false-positive generator. All of it is repo-local; none requires enterprise infrastructure.

- **Held-out session.** Reserve at least one recording session the agent cannot see until a hypothesis is *pre-registered* (statement, predicted effect, chosen test, correction method) in the lab notebook. Confirmation runs only against held-out data.
- **Multiple-comparison control, tracked.** The notebook counts every test the agent runs in a campaign; corrections (Benjamini–Hochberg FDR via `statsmodels.stats.multitest`, or Bonferroni) are applied automatically and the family of tests is recorded. This is the single most important guardrail when an agent can spawn many analyses.
- **Power / effect-size sanity.** Before proposing an analysis, the Hypothesizer estimates required N (e.g. `statsmodels.stats.power`) and flags under-powered designs — especially relevant given per-opponent event counts (cf. the `label_mode='group'` pooling that already exists for exactly this reason).
- **Assumptions stated.** Every agent-written analysis ships with its statistical assumptions written into the module docstring and echoed into the notebook.
- **Domain guardrails encoded as tests.** Extend the existing pattern (e.g. the self/target swap test in `tests/test_social_spatial_fields.py`) so agent-generated modules must pass domain-invariant checks, not just "it runs."

## 6. Human-in-the-loop + the lab notebook

**Interaction.** Two surfaces, no new stack required:
- *Conversational* — this environment already is one (agent + repo + shell + literature MCPs). Start here.
- *Dashboard* — a thin view alongside the existing Streamlit (`gui/app.py`) / Panel (`gui/interactive_app.py`) GUIs: a **hypothesis queue** (ranked, pending), **code diffs awaiting approval**, and **result comparisons across iterations**, with one-click "approve → run."

**Approval gates (hard requirements).** The agent may run cached/known analyses freely, but must get explicit scientist approval before: (a) writing/modifying repo code, (b) running against the held-out session, (c) anything touching `config/` or `.gui_cache/` deletion (both flagged "don't touch without asking" in `CLAUDE.md`).

**Lab notebook — the one genuinely new artifact.** An append-only record (natural fit for the existing `database/` + `habitat_pipeline.db`) capturing, per iteration: hypothesis text, params, git commit of the code that ran, dataset/session version, result metrics, figure paths, literature citations, the test family for multiple-comparison accounting, and the scientist's decision. This gives provenance, resumability, and a reviewable audit trail in one place.

## 7. Hardening layer (right-sized ops)

Adopted from the ops-focused design, trimmed to what a single lab needs. Add these once a prototype proves the loop is worth industrializing — not before.

**Worth doing:**
- **Reproducible environment.** The repo already has `environment.yml` / `pyproject.toml` / `pixi.toml` / `requirements.txt`. Pin them and build one container image (Docker locally; **Apptainer/Singularity** on Janelia HPC, since Docker usually isn't allowed on shared clusters). Tag every notebook entry with the image ID.
- **CI that gates agent code.** GitHub Actions (or Janelia's CI) on each agent PR: `ruff`/`flake8` + `mypy`, then `pytest` via `tests/run_tests.py` on mock data, then a smoke run of one small analysis. Agent-authored code **cannot merge** until this passes *and* a human approves the diff. This is the technical enforcement of principle #1.
- **Experiment tracking.** Log runs (params, metrics, artifacts) — MLflow (self-hosted, no vendor lock) or W&B if the lab already uses it. The `(session, params)` cache key is already most of a run identity; tracking just externalizes it next to the lab notebook.
- **Sandboxing.** Run agent-generated code in the container with no network and a scratch copy of the repo before any approved merge.

**Explicitly *not* now (avoid the enterprise trap):** Kubernetes / Terraform / cloud autoscaling, a managed platform (Domino), RLHF/fine-tuning the agent, and a bespoke web GUI. Each is weeks of work that buys little for one lab and can be revisited if usage grows across the group.

## 8. Phased roadmap

**Phase 0 — De-risk in this environment (days).** Run one full manual loop here on a real session (e.g. `--animal_id 631 --session_id 20251216`, and the `631/632` inter-brain pair). Goal: find where *your* data breaks the loop before building anything. Cheapest possible learning.

**Phase 1 — Codify the loop (1–2 weeks).** Author skills (`run-analysis`, `interpret-results`, `propose-hypotheses`, `implement-module`) and subagent definitions (Interpreter, Coder). Build the lab notebook schema on the existing `database/` layer. Wire in the rigor layer (§5). Still conversational; no web app.

**Phase 2 — Harden (2–4 weeks).** Container + CI gating on agent PRs + experiment tracking + sandboxing (§7). Now agent-written modules can be trusted through an automated gate.

**Phase 3 — Dashboard (optional, 2–4 weeks).** The hypothesis queue / review surface beside the existing GUIs, so others in the Tervo lab can use it, not just one person.

## 9. Corrections carried over from the source designs

For anyone reusing the earlier ops-focused report: the repo is **`habitat_pipeline`** (not `ratcity_pipeline`), and the modality is **extracellular electrophysiology** — Kilosort 4 spike data that is *already sorted* — with video tracking and manually-scored behavioral events. It is **not** calcium imaging: CaImAn, DeepLabCut, and "spike detection/filtering" preprocessing do not apply. And the "add extensibility hooks / module scaffolding / config-driven design" work is largely redundant — per-module CLIs, the unified result-dict schema, the `(session, params)` cache, the `database/` layer, and the mock-data test suite already exist and should be reused, not rebuilt.

## References

- FutureHouse Robin — https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system · Nature https://www.nature.com/articles/s41586-026-10652-y · code https://github.com/Future-House/robin/
- Google DeepMind Co-Scientist — https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/ · Nature https://www.nature.com/articles/s41586-026-10644-y
- The AI Scientist-v2 — https://arxiv.org/abs/2504.08066
- Evaluating AI agents on a neuroscience data-to-discovery pipeline — https://arxiv.org/pdf/2606.07718
