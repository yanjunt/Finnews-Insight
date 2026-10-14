"""Validate a saved classifier and hash the exact files used for deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory):
    directory = Path(directory).expanduser().resolve()
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise ValueError("The directory does not contain config.json.")
    config = json.loads(config_path.read_text())
    labels = {str(k): str(v).lower() for k, v in config.get("id2label", {}).items()}
    if set(labels.values()) != {"negative", "neutral", "positive"}:
        raise ValueError(f"Unexpected sentiment labels: {labels}")
    if len(labels) != 3:
        raise ValueError("Exactly three sentiment labels are required.")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError("tokenizer_config.json is missing.")
    if not any((directory / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        raise ValueError("No tokenizer vocabulary was found.")
    weight_paths = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
    if not weight_paths:
        raise ValueError("No saved model weights were found.")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "model_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"model_type": config.get("model_type"), "id2label": labels, "files": files,
            "note": "File hashes support identity checking; they do not prove model accuracy or training provenance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory")
    parser.add_argument("--output", type=Path, default=Path("model_manifest.json"))
    args = parser.parse_args()
    manifest = validate(args.model_directory)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved model validation passed.")
    print("Model type:", manifest["model_type"])
    print("Labels:", manifest["id2label"])
    print("Manifest:", args.output.resolve())


if __name__ == "__main__":
    main()
