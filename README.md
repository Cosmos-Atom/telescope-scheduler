---
title: Telescope Scheduler
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - scheduling
  - astronomy
---

# Telescope Scheduling Environment

An OpenEnv environment for autonomous exoplanet observation scheduling at **Mauna Kea Observatory** (Hawaii, 4,207 m).

An AI agent must decide which of 20 exoplanet targets to observe each night, balancing scientific priority, atmospheric quality, stochastic weather, and time-sensitive deadlines — mirroring the real trade-offs faced by professional observatories.

**This environment fills a real gap in the RL/agent benchmarking landscape**: most scheduling benchmarks use synthetic or game-like domains. Telescope scheduling uses actual NASA Exoplanet Archive data and astropy physics, making it a rigorous, grounded benchmark for evaluating LLM agents on real combinatorial resource allocation under uncertainty.

---

## Environment Overview

Each episode simulates one observing night (sunset to sunrise). At every 15-minute timestep the agent observes a planet (or waits), accumulating a reward based on:

- **Scientific priority** of the target (1–27 priority score derived from transit detectability)
- **Atmospheric quality** — altitude above horizon and airmass penalty
- **Weather state** — clear / partial clouds / bad (Markov chain)
- **Deadline urgency** — bonus for observing deadline targets on time, penalty for missing them

### Action Space

| Action | Description |
|--------|-------------|
| `0–19` | Observe planet at that index |
| `20` | Wait one 15-minute slot (small penalty) |

### Observation

Each step returns a `TelescopeObservation` with:
- Full LLM-readable `narrative` text listing all targets, altitudes, priorities, and deadlines
- Per-planet `PlanetInfo` structs (altitude, airmass, visibility, deadline status)
- Weather state, time remaining, step count

---

## Tasks

| Task ID | Description | Steps | Weather | Deadlines | Grader |
|---------|-------------|-------|---------|-----------|--------|
| `easy` | Clear night, full duration | 44 | Locked clear | None | `n_observed / 20` |
| `medium` | Partial night, variable weather | 32 | Stochastic | None | `priority_sum / 182` |
| `hard` | Short window, deadline pressure | 18 | Stochastic | Top-3 targets by step 5 | `0.6 × deadline + 0.4 × priority` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
PYTHONPATH=telescope_env/:telescope_env/server/ uvicorn telescope_env.server.app:app --port 7860 --reload
```

Or from inside `telescope_env/`:
```bash
PYTHONPATH=.:server/ uvicorn server.app:app --port 7860 --reload
```

### 3. Run inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-8B"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:7860"

python telescope_env/inference.py
```

Expected output (Qwen/Qwen3-8B baseline):
```
[START] task=easy env=telescope-scheduler model=Qwen/Qwen3-8B
[STEP] step=1 action=1 reward=+0.33 done=false error=null
...
[END] success=true steps=44 score=0.750 rewards=...

[START] task=medium env=telescope-scheduler model=Qwen/Qwen3-8B
...
[END] success=true steps=32 score=0.640 rewards=...

[START] task=hard env=telescope-scheduler model=Qwen/Qwen3-8B
...
[END] success=true steps=18 score=0.580 rewards=...
```

### 4. Docker

```bash
docker build -t telescope-scheduler telescope_env/
docker run -p 7860:7860 telescope-scheduler
```

---

## File Structure

```
telescope_env/
├── Dockerfile               # Root Dockerfile for HF Space / docker build
├── README.md
├── __init__.py
├── pyproject.toml           # Project metadata + [project.scripts] entry point
├── uv.lock                  # Locked dependencies (required by openenv)
├── requirements.txt
├── openenv.yaml             # OpenEnv spec manifest
├── models.py                # Pydantic: ObserveAction, TelescopeObservation, TelescopeState
├── client.py                # EnvClient subclass for Python API access
├── inference.py             # MANDATORY: OpenAI-client LLM baseline
├── outputs/                 # Episode output directory (required by openenv)
├── data/
│   ├── phase1_agent_observable.csv    # 20 exoplanet targets with RA/Dec
│   └── phase2_priority_durations.csv  # Priority scores and observation durations
└── server/
    ├── __init__.py
    ├── Dockerfile           # Alternative Dockerfile (server-only context)
    ├── app.py               # FastAPI app entry point
    ├── core.py              # Internal scheduling engine (_TelescopeCore)
    └── environment.py       # OpenEnv Environment subclass
```

---

## Scoring

Scores are computed client-side in `inference.py::compute_grade()`:

| Task | Formula | Oracle upper bound |
|------|---------|-------------------|
| easy | `min(n_observed_tonight / 20, 1.0)` | 1.000 (observe all 20 planets) |
| medium | `min(total_priority_observed / 182, 1.0)` | 1.000 (greedy priority policy) |
| hard | `0.6 × (deadlines_met / 3) + 0.4 × min(priority / 133, 1.0)` | 1.000 |

---

## Real-World Basis

Planet data is derived from the **NASA Exoplanet Archive** (confirmed transiting planets observable from Mauna Kea). Priority scores reflect transit signal-to-noise ratio and scheduling urgency — the same factors used by the Keck Observatory's actual scheduling committee.

Observatory coordinates: 19.82°N, 155.47°W, 4,207 m (Mauna Kea, Hawaii).
