"""Merge LoRA adapter into base model and optionally export GGUF."""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_and_save(
    base_model: str,
    adapter_path: str | Path,
    output_path: str | Path,
    max_seq_len: int = 2048,
    load_in_4bit: bool = False,
) -> dict:
    """Merge LoRA adapter into base model and save as 16-bit.

    Must run on GPU environment.
    """
    from unsloth import FastLanguageModel

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )

    return {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "output_path": str(output_path),
        "method": "merged_16bit",
    }


def measure_latency(
    model_path: str | Path,
    prompts: list[str],
    max_tokens: int = 512,
    max_seq_len: int = 2048,
) -> dict:
    """Measure inference latency on a merged model with vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        dtype="float16",
        max_model_len=max_seq_len,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "num_prompts": len(prompts),
        "total_seconds": round(elapsed, 2),
        "seconds_per_query": round(elapsed / len(prompts), 3),
        "total_output_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / elapsed, 1),
    }


def export_gguf(
    adapter_path: str | Path,
    output_path: str | Path,
    quantization: str = "q4_k_m",
    max_seq_len: int = 2048,
) -> dict:
    """Export merged model to GGUF format (optional)."""
    from unsloth import FastLanguageModel

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_len,
        dtype=None,
    )

    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=quantization,
    )

    return {
        "adapter_path": str(adapter_path),
        "output_path": str(output_path),
        "quantization": quantization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--export-gguf", action="store_true")
    parser.add_argument("--gguf-quant", default="q4_k_m")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = merge_and_save(
        args.base_model, args.adapter_path, args.output_path,
        args.max_seq_len, args.load_in_4bit,
    )
    print(json.dumps(result, indent=2))

    if args.export_gguf:
        gguf_path = args.output_path / "gguf"
        gguf_result = export_gguf(args.adapter_path, gguf_path, args.gguf_quant)
        print(json.dumps(gguf_result, indent=2))


if __name__ == "__main__":
    main()
