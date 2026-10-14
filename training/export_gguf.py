"""Optional CLI wrapper around the existing Unsloth GGUF exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.merge_adapter import export_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", default="q4_k_m")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    report = export_gguf(args.adapter, args.output, args.quantization, args.max_seq_len)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
