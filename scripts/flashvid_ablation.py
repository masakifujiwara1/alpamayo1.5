#!/usr/bin/env python3
"""Small Alpamayo baseline/FlashVID A/B runner.

The default run keeps the evaluation cheap: one clip and one front camera.
With ``--mode auto``, Alpamayo 1.5 runs VQA and Alpamayo R1 runs trajectory
because R1 does not expose the 1.5 VQA/text-generation helper.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPAMAYO15_ROOT = REPO_ROOT / "alpamayo1.5"
ALPAMAYO_R1_ROOT = REPO_ROOT / "alpamayo"
for src_root in (ALPAMAYO15_ROOT / "src", ALPAMAYO_R1_ROOT / "src"):
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import numpy as np
import torch


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_QUESTION = "Describe the scene."
DEFAULT_MODEL_IDS = {
    "alpamayo1.5": "nvidia/Alpamayo-1.5-10B",
    "alpamayo-r1": "nvidia/Alpamayo-R1-10B",
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    default_model_id: str
    model_cls: Any
    helper: Any
    load_dataset: Any
    supports_vqa: bool
    supports_camera_metadata: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-family",
        choices=("auto", "alpamayo1.5", "alpamayo-r1"),
        default="auto",
        help="Model code path to use. auto infers from --model-id when possible.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HF model id or local checkpoint. Defaults depend on --model-family.",
    )
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--mode", choices=("auto", "vqa", "traj", "both"), default="auto")
    parser.add_argument("--camera-set", choices=("front-only", "default"), default="front-only")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--flashvid-path", type=Path, default=REPO_ROOT / "candidate" / "FlashVID")
    parser.add_argument("--token-selection-method", default="attn")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temporal-threshold", type=float, default=1.0)
    parser.add_argument("--do-segment", action="store_true")
    parser.add_argument("--segment-threshold", type=float, default=0.9)
    parser.add_argument("--min-segment-num", type=int, default=4)
    parser.add_argument("--no-complementary-segment", action="store_true")
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--pruning-layer", type=int, default=28)
    parser.add_argument("--llm-retention-ratio", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def infer_model_family(model_family: str, model_id: str | None) -> str:
    if model_family != "auto":
        return model_family
    if model_id is not None:
        normalized = model_id.lower()
        if "alpamayo-r1" in normalized or "alpamayo_r1" in normalized:
            return "alpamayo-r1"
        if "alpamayo-1.5" in normalized or "alpamayo1.5" in normalized:
            return "alpamayo1.5"
    return "alpamayo1.5"


def load_model_family(name: str) -> ModelFamily:
    if name == "alpamayo-r1":
        from alpamayo_r1 import helper
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        return ModelFamily(
            name=name,
            default_model_id=DEFAULT_MODEL_IDS[name],
            model_cls=AlpamayoR1,
            helper=helper,
            load_dataset=load_physical_aiavdataset,
            supports_vqa=False,
            supports_camera_metadata=False,
        )

    if name == "alpamayo1.5":
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        return ModelFamily(
            name=name,
            default_model_id=DEFAULT_MODEL_IDS[name],
            model_cls=Alpamayo1_5,
            helper=helper,
            load_dataset=load_physical_aiavdataset,
            supports_vqa=True,
            supports_camera_metadata=True,
        )

    raise ValueError(f"Unsupported model family: {name}")


def resolve_mode(mode: str, family: ModelFamily) -> str:
    if mode == "auto":
        return "vqa" if family.supports_vqa else "traj"
    if mode in ("vqa", "both") and not family.supports_vqa:
        raise ValueError(
            f"{family.name} does not support VQA mode in this runner. Use --mode traj."
        )
    return mode


def camera_features(args: argparse.Namespace, avdi: Any) -> list[Any] | None:
    if args.camera_set == "default":
        return None
    return [avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def tokenized_summary(tokenized_data: dict[str, Any], merge_size: int) -> dict[str, Any]:
    image_grid_thw = tokenized_data.get("image_grid_thw")
    if image_grid_thw is None:
        tokens_per_image = []
        total_visual_tokens = 0
    else:
        grid = image_grid_thw.detach().cpu()
        tokens = grid.prod(-1) // (merge_size**2)
        tokens_per_image = [int(item) for item in tokens.tolist()]
        total_visual_tokens = int(tokens.sum().item())

    return {
        "input_ids": tensor_summary(tokenized_data.get("input_ids")),
        "attention_mask": tensor_summary(tokenized_data.get("attention_mask")),
        "pixel_values": tensor_summary(tokenized_data.get("pixel_values")),
        "image_grid_thw": to_jsonable(image_grid_thw),
        "tokens_per_image": tokens_per_image,
        "total_visual_tokens": total_visual_tokens,
        "num_images": len(tokens_per_image),
    }


def prepare_vqa_inputs(
    family: ModelFamily,
    processor: Any,
    data: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = family.helper.create_vqa_message(
        data["image_frames"].flatten(0, 1),
        question=args.question,
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    summary = tokenized_summary(inputs, merge_size)
    return family.helper.to_device({"tokenized_data": inputs}, device), summary


def prepare_traj_inputs(
    family: ModelFamily,
    processor: Any,
    data: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = data["image_frames"].flatten(0, 1)
    if family.supports_camera_metadata:
        messages = family.helper.create_message(
            frames=frames,
            camera_indices=data["camera_indices"],
        )
    else:
        messages = family.helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    summary = tokenized_summary(inputs, merge_size)
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return family.helper.to_device(model_inputs, device), summary


def set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def fresh_model_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    """Copy the outer containers so model calls can pop input_ids safely."""
    copied = dict(model_inputs)
    if "tokenized_data" in copied:
        copied["tokenized_data"] = dict(copied["tokenized_data"])
    return copied


@contextlib.contextmanager
def autocast_for(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        yield


def measure(label: str, device: str, fn: Any) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    payload = fn()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_memory_gb = None

    return {
        "label": label,
        "latency_sec": time.perf_counter() - start,
        "peak_memory_gb": peak_memory_gb,
        **payload,
    }


def flashvid_config_summary(model: Any) -> dict[str, Any] | None:
    config = getattr(model.vlm, "flashvid_config", None)
    if config is None:
        return None
    keys = (
        "compression_source",
        "retention_ratio",
        "original_visual_token_length",
        "vision_side_visual_token_length",
        "visual_token_length",
        "num_attn_div_tokens",
        "num_sttm_tokens",
        "pruning_layer",
        "llm_retention_ratio",
    )
    return {key: getattr(config, key, None) for key in keys}


def run_vqa(
    model: Any,
    model_inputs: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_seed(args.seed, args.device)
    with torch.no_grad(), autocast_for(args.device, dtype):
        extra = model.generate_text(
            data=fresh_model_inputs(model_inputs),
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_samples=args.num_samples,
            max_generation_length=args.max_generation_length,
        )
    return {"extra": to_jsonable(extra), "flashvid": flashvid_config_summary(model)}


def run_traj(
    model: Any,
    model_inputs: dict[str, Any],
    raw_data: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_seed(args.seed, args.device)
    with torch.no_grad(), autocast_for(args.device, dtype):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=fresh_model_inputs(model_inputs),
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    gt_xy = raw_data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return {
        "minADE": float(diff.min()),
        "extra": to_jsonable(extra),
        "flashvid": flashvid_config_summary(model),
    }


def apply_flashvid(model: Any, args: argparse.Namespace, ratio: float) -> None:
    sys.path.insert(0, str(args.flashvid_path))
    from flashvid import flashvid

    model.vlm = flashvid(
        model.vlm,
        retention_ratio=ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=not args.no_complementary_segment,
        token_selection_method=args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        expansion=args.expansion,
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
    )


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    family_name = infer_model_family(args.model_family, args.model_id)
    family = load_model_family(family_name)
    args.model_id = args.model_id or family.default_model_id
    args.mode = resolve_mode(args.mode, family)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data = family.load_dataset(
        args.clip_id,
        t0_us=args.t0_us,
        avdi=avdi,
        camera_features=camera_features(args, avdi),
    )

    model = family.model_cls.from_pretrained(args.model_id, dtype=dtype).to(args.device)
    model.eval()
    processor = family.helper.get_processor(model.tokenizer)

    results: dict[str, Any] = {
        "config": {
            "model_family": family.name,
            "model_id": args.model_id,
            "clip_id": args.clip_id,
            "t0_us": args.t0_us,
            "question": args.question,
            "mode": args.mode,
            "camera_set": args.camera_set,
            "ratios": args.ratios,
            "flashvid_path": str(args.flashvid_path),
        },
        "tokenized": {},
        "runs": [],
    }

    vqa_inputs = traj_inputs = None
    if args.mode in ("vqa", "both"):
        vqa_inputs, results["tokenized"]["vqa"] = prepare_vqa_inputs(
            family, processor, data, args, args.device
        )
    if args.mode in ("traj", "both"):
        traj_inputs, results["tokenized"]["traj"] = prepare_traj_inputs(
            family, processor, data, args.device
        )

    if vqa_inputs is not None:
        results["runs"].append(
            measure("baseline_vqa", args.device, lambda: run_vqa(model, vqa_inputs, args, dtype))
        )
    if traj_inputs is not None:
        results["runs"].append(
            measure(
                "baseline_traj",
                args.device,
                lambda: run_traj(model, traj_inputs, data, args, dtype),
            )
        )

    for ratio in args.ratios:
        apply_flashvid(model, args, ratio)
        if vqa_inputs is not None:
            results["runs"].append(
                measure(
                    f"flashvid_r{ratio:g}_vqa",
                    args.device,
                    lambda: run_vqa(model, vqa_inputs, args, dtype),
                )
            )
        if traj_inputs is not None:
            results["runs"].append(
                measure(
                    f"flashvid_r{ratio:g}_traj",
                    args.device,
                    lambda: run_traj(model, traj_inputs, data, args, dtype),
                )
            )

    rendered = json.dumps(to_jsonable(results), indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
