"""Download only the pinned checkpoint; normal evaluation remains offline."""

from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
revision = (root / "model-revision.txt").read_text().strip()

if __name__ == "__main__":
    print(snapshot_download(
        repo_id="Qwen/Qwen3-0.6B", revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    ))
