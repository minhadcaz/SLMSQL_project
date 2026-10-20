"""Lazy vLLM backend. A command loads one checkpoint only when work is pending."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from research.slm_sql.artifacts import file_hash
from research.generation_metrics import completion_logprob


def model_identity(model: str, revision: str) -> dict:
    if Path(model).is_dir():
        root = Path(model).resolve()
        files = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".jinja", ".txt")
        )
        if not any(p.suffix in (".safetensors", ".bin") for p in files):
            raise ValueError(f"No model weights in {root}")
        return {
            "model": str(root),
            "revision": "local",
            "files": {p.name: file_hash(p) for p in files},
        }
    from huggingface_hub import model_info

    info = model_info(model, revision=revision)
    return {"model": model, "revision": info.sha}


class VLLMBackend:
    @staticmethod
    def runtime_versions():
        return {
            name: importlib.metadata.version(name) for name in ("vllm", "transformers", "torch")
        }

    def __init__(
        self, settings: dict, *, frozen_identity: dict | None = None, adapter: str | None = None
    ):
        from vllm import LLM, SamplingParams

        self.identity = frozen_identity or model_identity(settings["model"], settings["revision"])
        if self.identity["model"] != settings["model"] and not Path(settings["model"]).is_dir():
            raise ValueError("Cached model identity differs from configured checkpoint.")
        if frozen_identity and Path(settings["model"]).is_dir():
            if model_identity(settings["model"], settings["revision"]) != frozen_identity:
                raise ValueError("Local model weights changed after the artifact was created.")
        self.settings = settings
        self.adapter = None
        kwargs = {}
        if adapter:
            from vllm.lora.request import LoRARequest
            from research.slm_sql.artifacts import read_json

            rank = read_json(Path(adapter) / "adapter_config.json")["r"]
            kwargs.update(enable_lora=True, max_lora_rank=max(8, rank))
            self.adapter = LoRARequest("csc-adapted", 1, adapter)
        revision = self.identity["revision"]
        self.llm = LLM(
            model=self.identity["model"],
            revision=None if revision == "local" else revision,
            tokenizer_revision=None if revision == "local" else revision,
            dtype=settings["dtype"],
            max_model_len=settings["max_model_len"],
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params_class = SamplingParams
        self.versions = self.runtime_versions()

    def generate(self, messages: list[list[dict]], *, n: int, seeds: list[int]) -> list[list[dict]]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        budget = self.settings["max_model_len"] - self.settings["max_tokens"]
        if any(length > budget for length in lengths):
            raise ValueError(
                f"Prompt exceeds context budget: max={max(lengths)}, available={budget}. "
                "Increase max_model_len; prompts are never silently truncated."
            )
        params = [
            self.params_class(
                n=n,
                temperature=self.settings["temperature"],
                top_p=self.settings["top_p"],
                max_tokens=self.settings["max_tokens"],
                seed=seed,
                logprobs=1,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(prompts, params, lora_request=self.adapter, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned a different number of requests.")
        result = []
        for output, prompt_tokens in zip(outputs, lengths):
            if len(output.outputs) != n:
                raise RuntimeError(f"Expected {n} completions, got {len(output.outputs)}.")
            result.append(
                [
                    {
                        "raw_text": completion.text,
                        "n_tokens": len(completion.token_ids),
                        "avg_logprob": completion_logprob(completion),
                        "finish_reason": completion.finish_reason,
                        "prompt_tokens": prompt_tokens,
                    }
                    for completion in output.outputs
                ]
            )
        return result
