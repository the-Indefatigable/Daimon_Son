"""Bedrock Llama drafter. Generates N post candidates from a mix of models
so Claude-judge has a diverse slate to pick from.

Design: 70B for quality baseline, Scout 17B for fine-tune path (same model ID
swaps to the fine-tuned variant once trained). Each draft is tagged with its
source model so pick-rate becomes the first learning-loop signal.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
from botocore.config import Config as BotoConfig

REGION = os.getenv("AWS_REGION", "us-east-2")

MODEL_LLAMA_70B = "us.meta.llama3-3-70b-instruct-v1:0"
MODEL_SCOUT_17B = "us.meta.llama4-scout-17b-instruct-v1:0"

# (model_id, count) — default draft mix. 2+2 gives Claude a diverse slate
# without inflating judge-call latency.
DEFAULT_MIX: list[tuple[str, int]] = [
    (MODEL_LLAMA_70B, 2),
    (MODEL_SCOUT_17B, 2),
]

# Per-1M-token Bedrock pricing (us-east-2, on-demand). Keep in sync with
# https://aws.amazon.com/bedrock/pricing/
BEDROCK_PRICING = {
    MODEL_LLAMA_70B: {"input": 0.72, "output": 0.72},
    MODEL_SCOUT_17B: {"input": 0.17, "output": 0.66},
}

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 300


@dataclass
class Draft:
    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "standard"},
                read_timeout=60,
            ),
        )
    return _client


def _cost(model_id: str, in_tok: int, out_tok: int) -> float:
    p = BEDROCK_PRICING.get(model_id)
    if not p:
        return 0.0
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


def _one_draft(
    model_id: str,
    prompt: str,
    system: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Draft:
    client = _get_client()
    kwargs = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }
    if system:
        kwargs["system"] = [{"text": system}]

    t0 = time.time()
    resp = client.converse(**kwargs)
    latency_ms = int((time.time() - t0) * 1000)

    text = resp["output"]["message"]["content"][0]["text"].strip()
    usage = resp.get("usage", {})
    in_tok = int(usage.get("inputTokens", 0))
    out_tok = int(usage.get("outputTokens", 0))
    return Draft(
        text=text,
        model_id=model_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=_cost(model_id, in_tok, out_tok),
    )


def draft(
    prompt: str,
    *,
    system: str | None = None,
    mix: list[tuple[str, int]] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Draft]:
    """Generate drafts concurrently from the configured model mix.

    Returns a list of Draft objects in completion order. Order is irrelevant —
    Claude-judge receives them as a shuffled slate.
    """
    mix = mix or DEFAULT_MIX
    jobs: list[str] = []
    for model_id, count in mix:
        jobs.extend([model_id] * count)

    drafts: list[Draft] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(_one_draft, model_id, prompt, system, temperature, top_p, max_tokens)
            for model_id in jobs
        ]
        for fut in as_completed(futures):
            drafts.append(fut.result())
    return drafts
