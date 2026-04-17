"""Smoke test: hit Bedrock Llama models via Converse API.

Usage:
    .venv/bin/python scripts/bedrock_smoke.py          # test all three
    .venv/bin/python scripts/bedrock_smoke.py llama3   # test just 8B
"""
from __future__ import annotations
import sys
import time
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-2"

MODELS = {
    "llama33": "us.meta.llama3-3-70b-instruct-v1:0",
    "llama4":  "us.meta.llama4-scout-17b-instruct-v1:0",
    "llama3":  "us.meta.llama3-1-8b-instruct-v1:0",
}

PROMPT = "In one sentence, describe a feral raccoon running a fintech startup."


def test_model(client, label: str, model_id: str) -> None:
    print(f"\n=== {label} ({model_id}) ===")
    t0 = time.time()
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.9},
        )
    except ClientError as e:
        print(f"  ERROR: {e.response['Error']['Code']} — {e.response['Error']['Message']}")
        return

    elapsed = time.time() - t0
    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    print(f"  {text.strip()}")
    print(f"  ({elapsed:.1f}s | in={usage.get('inputTokens')} out={usage.get('outputTokens')})")


def main() -> None:
    client = boto3.client("bedrock-runtime", region_name=REGION)

    pick = sys.argv[1] if len(sys.argv) > 1 else None
    if pick and pick in MODELS:
        test_model(client, pick, MODELS[pick])
    else:
        for label, model_id in MODELS.items():
            test_model(client, label, model_id)


if __name__ == "__main__":
    main()
