"""Pinned external HuBERT phone classifier, with no installed torchaudio dependency.

The publisher trained the checkpoint on LibriSpeech train-clean-100. Loading uses
unmodified, pinned pure-PyTorch torchaudio definitions and the publisher's exact
inference architecture. No L2-ARCTIC data train this model. This is a frame phone
classifier, not an end-to-end mispronunciation detector or a CTC model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

MODEL_ID = "coml/hubert-phoneme-classification"
MODEL_REVISION = "9251d0fdc406d973ce1dbb7738ac6f7b124fbc65"
AUTHOR_REVISION = "47225179e664787b28e78edb0ff77e20a82a0078"
AUDIO_TAG = "v2.8.0"
AUDIO_REVISION = "6e1c7fe9ff6d82b8665d0a46d859d3357d2ebaaa"
MODEL_SHA256 = "8c06c6a0e45bd6cfebfac66b2ac7a7111a692a96682c09d0cc715b6e710c868d"
MODEL_BYTES = 378939152
PHONEMES = ("SIL", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH",
            "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG",
            "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH")
PHONE_TO_ID = {phone: i for i, phone in enumerate(PHONEMES)}
CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
FRAME_STRIDE = 320
RECEPTIVE_FIELD = 400
SAMPLE_RATE = 16000
AUTHOR_CONFIG = {
    "encoder_projection_dropout": 0, "encoder_attention_dropout": 0,
    "encoder_ff_interm_dropout": .1, "encoder_dropout": 0, "encoder_layer_drop": .1,
    "mask_prob": .75, "mask_channel_prob": .5, "mask_channel_length": 10, "num_classes": 500,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def conv_output_length(samples: int) -> int:
    if not isinstance(samples, int) or samples < RECEPTIVE_FIELD:
        raise ValueError("audio must contain at least 400 samples")
    for kernel, stride in zip(CONV_KERNELS, CONV_STRIDES):
        samples = (samples - kernel) // stride + 1
    return samples


def frame_center_samples(n_frames: int):
    """Geometric centers of raw convolution receptive fields; no learned latency claim."""
    import numpy as np
    if not isinstance(n_frames, int) or n_frames < 0:
        raise ValueError("nonnegative integer frame count required")
    return (RECEPTIVE_FIELD - 1) / 2 + FRAME_STRIDE * np.arange(n_frames, dtype=float)


def fetch_assets(destination: Path) -> dict:
    """Fetch only publisher-pinned safetensors and small, licensed source definitions."""
    import requests
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = destination / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        verify_assets(destination, provenance)
        return provenance
    session = requests.Session()
    def get_json(url):
        response = session.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
    # Resolve the annotated tag to its immutable commit before any source download.
    ref = get_json(f"https://api.github.com/repos/pytorch/audio/git/ref/tags/{AUDIO_TAG}")["object"]
    while ref["type"] == "tag":
        ref = get_json(f"https://api.github.com/repos/pytorch/audio/git/tags/{ref['sha']}")["object"]
    if ref["type"] != "commit":
        raise ValueError("torchaudio tag did not resolve to a commit")
    audio_revision = ref["sha"]
    if audio_revision != AUDIO_REVISION:
        raise ValueError("publisher torchaudio tag no longer matches the pinned commit")
    tree = get_json(f"https://huggingface.co/api/models/{MODEL_ID}/tree/{MODEL_REVISION}")
    model_entry = next(row for row in tree if row["path"] == "model.safetensors")
    expected_model_sha = model_entry["lfs"]["oid"]
    expected_model_bytes = model_entry["lfs"]["size"]
    if expected_model_sha != MODEL_SHA256 or expected_model_bytes != MODEL_BYTES:
        raise ValueError("publisher checkpoint metadata differs from pinned LFS identity")
    sources = {
        "README.md": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/README.md",
        "author/model.py": f"https://raw.githubusercontent.com/bootphon/spokenlm-phoneme/{AUTHOR_REVISION}/src/phonslm/model.py",
        "author/data.py": f"https://raw.githubusercontent.com/bootphon/spokenlm-phoneme/{AUTHOR_REVISION}/src/phonslm/data.py",
        "author/LICENSE": f"https://raw.githubusercontent.com/bootphon/spokenlm-phoneme/{AUTHOR_REVISION}/LICENSE",
        "torchaudio/LICENSE": f"https://raw.githubusercontent.com/pytorch/audio/{audio_revision}/LICENSE",
    }
    for name in ("model.py", "components.py", "wavlm_attention.py"):
        sources[f"torchaudio/{name}"] = f"https://raw.githubusercontent.com/pytorch/audio/{audio_revision}/src/torchaudio/models/wav2vec2/{name}"
    sources["model.safetensors"] = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/model.safetensors"
    files = {}
    for relative, url in sources.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if relative == "model.safetensors" and sha256(path) == expected_model_sha and path.stat().st_size == expected_model_bytes:
                pass
            elif relative != "model.safetensors":
                response = session.get(url, timeout=60)
                response.raise_for_status()
                if path.read_bytes() != response.content:
                    raise ValueError(f"refusing to replace existing source: {relative}")
            else:
                raise ValueError("existing checkpoint differs from publisher LFS hash")
        else:
            partial = path.with_name(path.name + ".part")
            if partial.exists():
                raise ValueError(f"incomplete prior download requires review: {relative}")
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            if relative == "model.safetensors":
                if partial.stat().st_size != expected_model_bytes or sha256(partial) != expected_model_sha:
                    raise ValueError("checkpoint does not match pinned publisher LFS object")
            partial.rename(path)
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source_url": url}
        print(json.dumps({"downloaded_or_verified": relative, "bytes": path.stat().st_size}), flush=True)
    provenance = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "author_revision": AUTHOR_REVISION,
        "torchaudio_tag": AUDIO_TAG, "torchaudio_revision": audio_revision,
        "training_source": "publisher model card: LibriSpeech train-clean-100; no L2-ARCTIC training declared",
        "model_license": "MIT per publisher model card and author LICENSE",
        "torchaudio_license": "BSD-2-Clause per retained upstream LICENSE",
        "publisher_lfs_sha256": expected_model_sha, "files": files,
        "phone_vocabulary_in_author_order": list(PHONEMES), "sample_rate": SAMPLE_RATE,
        "constructor_configuration": {"freeze_encoder": True, "ctc_training": False,
            "source": "author constructor defaults; publisher main has no config.json; 40-class checkpoint shapes and strict load verify frame-classification architecture"},
        "inference_implementation": "unmodified torchaudio pure-PyTorch definitions, author inference architecture, full strict checkpoint loading; omit only redundant initial base-checkpoint download",
        "limits": ["Frame classification, not CTC", "No original author dependency environment reproduced", "Native forced-alignment supervision is not independent human phonetic gold", "No target-L2 performance checked during model choice"],
    }
    save_json(provenance_path, provenance)
    verify_assets(destination, provenance)
    return provenance


def verify_assets(destination: Path, provenance: dict | None = None) -> dict:
    if provenance is None:
        provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
    if provenance["model_id"] != MODEL_ID or provenance["model_revision"] != MODEL_REVISION or provenance["author_revision"] != AUTHOR_REVISION:
        raise ValueError("pinned external model identity changed")
    if provenance["torchaudio_revision"] != AUDIO_REVISION or provenance["publisher_lfs_sha256"] != MODEL_SHA256:
        raise ValueError("pinned upstream source/checkpoint identity changed")
    if provenance["phone_vocabulary_in_author_order"] != list(PHONEMES):
        raise ValueError("publisher token order changed")
    for relative, expected in provenance["files"].items():
        path = destination / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"external model/source hash changed: {relative}")
    if provenance["files"]["model.safetensors"]["sha256"] != provenance["publisher_lfs_sha256"]:
        raise ValueError("checkpoint does not match pinned LFS object")
    return provenance


def load_model(destination: Path):
    """Return an eval-only model: model(waveform[batch,samples]) -> [batch,frames,40]."""
    import torch
    from safetensors.torch import load_file
    provenance = verify_assets(destination)
    package_name = "_vpa_pinned_torchaudio_" + provenance["torchaudio_revision"][:12]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(destination / "torchaudio")]
        sys.modules[package_name] = package
        for name in ("wavlm_attention", "components", "model"):
            qualified = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(qualified, destination / "torchaudio" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
    source = sys.modules[f"{package_name}.model"]
    class ExternalPhoneClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = source.hubert_pretrain_base(**AUTHOR_CONFIG)
            self.aux = torch.nn.Linear(768, len(PHONEMES))
        def forward(self, waveforms):
            if waveforms.ndim != 2 or waveforms.shape[1] < RECEPTIVE_FIELD or not bool(torch.isfinite(waveforms).all()):
                raise ValueError("finite [batch,samples>=400] audio required")
            features, _ = self.model.wav2vec2(waveforms, None)
            return self.aux(features)
    model = ExternalPhoneClassifier()
    state = load_file(str(destination / "model.safetensors"), device="cpu")
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError("incomplete model state")
    model.eval().requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--synthetic-verify", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch_assets(args.destination)
    if args.synthetic_verify:
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        model = load_model(args.destination)
        # Synthetic audio verifies architecture only; it is not a speech metric.
        x = torch.linspace(-.1, .1, SAMPLE_RATE)[None, :]
        with torch.inference_mode():
            a, b = model(x), model(x)
        result = {"input": "synthetic deterministic ramp; no corpus audio",
                  "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "n_state_keys_strictly_loaded": len(model.state_dict()), "shape": list(a.shape),
                  "output_length_matches_convolution": a.shape[1] == conv_output_length(x.shape[1]),
                  "finite_logits": bool(torch.isfinite(a).all()), "exact_repeat_output": bool(torch.equal(a, b)),
                  "torch": torch.__version__, "threads": torch.get_num_threads(),
                  "frame_stride_samples": FRAME_STRIDE, "receptive_field_samples": RECEPTIVE_FIELD,
                  "first_geometric_center_samples": (RECEPTIVE_FIELD - 1) / 2,
                  "provenance_sha256": sha256(args.destination / "provenance.json")}
        save_json(args.destination / "synthetic_verification.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
