"""
Inference Script — Telescope Scheduling Environment
====================================================
MANDATORY

- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the
  root directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.
"""
import os
import re
import sys
import textwrap
from typing import List

# Ensure telescope_env package imports resolve when script is run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from client import TelescopeEnv
from models import ObserveAction

API_BASE_URL = os.getenv("API_BASE_URL")  # "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 44
TEMPERATURE = 0.2
MAX_TOKENS = 1024    # allow reasoning model <think> preamble before the integer
FALLBACK_TARGET = 20  # wait (num_planets = 20, so wait = index 20)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous telescope scheduler at Mauna Kea Observatory.
    Each turn you receive the current sky state and must choose one target to observe.
    Reply with ONLY a single integer:
      0-19 = observe that planet (by index shown in the observation)
      20   = wait one 15-minute slot
    No explanation. Just the number. /no_think
""").strip()

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def parse_target(response_text: str) -> int:
    # Strip thinking tags (Qwen3 and other reasoning models wrap output in <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", response_text or "", flags=re.DOTALL).strip()
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return max(0, min(20, int(match.group(1))))
    return FALLBACK_TARGET


def compute_grade(task_id: str, state) -> float:
    """Client-side grader — mirrors TelescopeSchedulingEnvironment.compute_grade() server-side.

    Denominators (oracle upper bounds reproducible via scripts/compute_oracle.py):
      easy   — 20 planets total in dataset
      medium — 182 = max priority sum achievable in 32 steps (greedy, seed=7)
      hard   — deadline_score: 3 soft-deadline targets
               priority_score: 133 = oracle value for 18 steps greedy-deadline policy
    """
    _EPS = 1e-4  # scores must be strictly within (0, 1) per OpenEnv validator
    if task_id == "easy":
        raw = min(state.n_observed_tonight / 20.0, 1.0)
    elif task_id == "medium":
        raw = min(state.total_priority_observed / 182.0, 1.0)
    else:  # hard
        deadline_score = min(getattr(state, "deadlines_met_before_cutoff", 0) / 3.0, 1.0)
        priority_score = min(state.total_priority_observed / 133.0, 1.0)
        raw = 0.6 * deadline_score + 0.4 * priority_score
    return round(max(_EPS, min(raw, 1.0 - _EPS)), 4)


DEBUG = True
BENCHMARK = "telescope-scheduler"
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error) -> None:
    error_val = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_id: str) -> float:
    _EPS = 1e-4
    rewards: List[float] = []
    steps_taken = 0
    score = _EPS  # never 0.0 — validator requires score strictly in (0, 1)

    log_start(task=task_id, model=MODEL_NAME or "unknown")
    try:
        with TelescopeEnv(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(task_id=task_id)

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs = result.observation
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": obs.narrative},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    raw = response.choices[0].message.content or ""
                    error = None
                except Exception as exc:
                    raw = ""
                    error = str(exc)

                target = parse_target(raw)
                result = env.step(ObserveAction(target=target, task_id=task_id))
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=target, reward=reward, done=result.done, error=error)

            try:
                final_state = env.state()
                score = compute_grade(task_id, final_state)
            except Exception as exc:
                print(f"[WARN] env.state() failed: {exc} — using fallback score", flush=True)
                score = _EPS
    finally:
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)


if __name__ == "__main__":
    main()
