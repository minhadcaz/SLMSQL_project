"""Download weights once, then inference can operate with a local cache."""
import argparse
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="cycloneboy/SLM-SQL-1.5B")
    parser.add_argument("--local-dir", type=Path)
    args = parser.parse_args()
    path = snapshot_download(
        args.model_id,
        local_dir=args.local_dir or Path(__file__).resolve().parents[1] / "models" / args.model_id.split("/")[-1],
        cache_dir=str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "hub"),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    print(path)


if __name__ == "__main__":
    main()
