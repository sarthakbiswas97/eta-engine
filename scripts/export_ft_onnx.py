"""Export FT-Transformer to ONNX for faster CPU inference.

ONNX Runtime fuses operations (LayerNorm + matmul, attention) and
eliminates Python dispatch overhead, giving 2-4x speedup for batch=1
small transformers.

Usage:
    PYTHONPATH=. python scripts/export_ft_onnx.py
    PYTHONPATH=. python scripts/export_ft_onnx.py --verify
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from model.ft_transformer import FTTransformer, FTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def export(ft_path: Path, onnx_path: Path) -> None:
    """Export FT-Transformer checkpoint to ONNX."""
    # Load PyTorch model
    checkpoint = torch.load(ft_path, map_location="cpu", weights_only=False)
    config = FTConfig(**checkpoint["model_config"])
    model = FTTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Loaded FT-Transformer: %d params", model.count_parameters())

    # Sample inputs for tracing
    batch = 1
    x_num = torch.randn(batch, config.n_numerical)
    x_cat = torch.randint(1, 266, (batch, config.n_categorical))

    # Export
    logger.info("Exporting to ONNX: %s", onnx_path)
    torch.onnx.export(
        model,
        (x_num, x_cat),
        str(onnx_path),
        input_names=["x_numerical", "x_categorical"],
        output_names=["prediction"],
        dynamic_axes={
            "x_numerical": {0: "batch"},
            "x_categorical": {0: "batch"},
            "prediction": {0: "batch"},
        },
        opset_version=17,
    )

    onnx_size = onnx_path.stat().st_size / 1e6
    logger.info("Exported ONNX model: %.1f MB", onnx_size)

    # Copy norm_params into a separate file for ONNX inference
    norm_path = ROOT / "ft_norm_params.npz"
    np.savez(
        norm_path,
        means=checkpoint["norm_params"]["means"],
        stds=checkpoint["norm_params"]["stds"],
    )
    logger.info("Saved normalization params: %s", norm_path)


def verify(ft_path: Path, onnx_path: Path) -> None:
    """Verify ONNX output matches PyTorch output."""
    import onnxruntime as ort

    # Load PyTorch model
    checkpoint = torch.load(ft_path, map_location="cpu", weights_only=False)
    config = FTConfig(**checkpoint["model_config"])
    model = FTTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ONNX model
    session = ort.InferenceSession(str(onnx_path))

    # Test with random inputs
    x_num = torch.randn(1, config.n_numerical)
    x_cat = torch.randint(1, 266, (1, config.n_categorical))

    # PyTorch prediction
    with torch.inference_mode():
        pt_pred = model(x_num, x_cat).item()

    # ONNX prediction
    ort_pred = session.run(
        ["prediction"],
        {
            "x_numerical": x_num.numpy(),
            "x_categorical": x_cat.numpy(),
        },
    )[0].item()

    diff = abs(pt_pred - ort_pred)
    logger.info("PyTorch: %.4f, ONNX: %.4f, diff: %.6f", pt_pred, ort_pred, diff)

    if diff < 0.01:
        logger.info("PASS: outputs match (diff < 0.01)")
    else:
        logger.warning("FAIL: outputs differ by %.4f", diff)

    # Benchmark
    import time

    # PyTorch
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(200):
            model(x_num, x_cat)
        pt_time = (time.perf_counter() - t0) / 200 * 1000

    # ONNX
    x_num_np = x_num.numpy()
    x_cat_np = x_cat.numpy()
    t0 = time.perf_counter()
    for _ in range(200):
        session.run(["prediction"], {"x_numerical": x_num_np, "x_categorical": x_cat_np})
    ort_time = (time.perf_counter() - t0) / 200 * 1000

    speedup = pt_time / ort_time
    logger.info("PyTorch: %.1fms, ONNX: %.1fms, speedup: %.1fx", pt_time, ort_time, speedup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FT-Transformer to ONNX")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX matches PyTorch + benchmark")
    args = parser.parse_args()

    ft_path = ROOT / "ft_model.pt"
    onnx_path = ROOT / "ft_model.onnx"

    if not ft_path.exists():
        raise SystemExit(f"Missing {ft_path}. Train FT-Transformer first.")

    export(ft_path, onnx_path)

    if args.verify:
        verify(ft_path, onnx_path)


if __name__ == "__main__":
    main()
