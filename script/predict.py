r"""Predict Delta G from one complex PDB using a single model or five-model ensemble.

    python predict.py --pdb complex.pdb --receptor-chains A --ligand-chains B \
        --output-dir outputs/complex

ESM2 defaults to the local torch hub cache. NUV always runs on CPU.
Weights default to model/model.ckpt beside this script.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import dgl
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from trigeo_inference import build_model, load_snapshot


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(checkpoint, config_snapshot=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_source = config_snapshot or payload.get("config") or ROOT / "configs/config.json"
    config, node_input_dim, label_stats = load_snapshot(config_source)
    if "models" in payload:
        bundle = payload
    else:
        name = checkpoint.name
        state_dict = payload.get("state_dict", payload)
        bundle = {
            "models": {name: {"state_dict": state_dict}},
            "ensemble_order": [name],
            "ensemble_weights": {name: 1.0},
        }
    config_origin = "checkpoint:config" if isinstance(config_source, dict) else str(config_source)
    return bundle, config, node_input_dim, label_stats, config_origin


def predict_ensemble(graphs, config, node_input_dim, label_stats, bundle, device):
    """Denormalize each checkpoint output, then take the arithmetic mean."""
    mean, std = float(label_stats["mean"]), float(label_stats["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Training label mean/std must be finite, with std > 0")
    graph1, graph2 = [graph.to(device) for graph in graphs]
    model_names = bundle["ensemble_order"]
    if len(model_names) not in (1, 5) or len(set(model_names)) != len(model_names) or set(model_names) != set(bundle["models"]):
        raise ValueError("Expected one model or five distinct ensemble models")
    if any(bundle["ensemble_weights"][name] != 1.0 / len(model_names) for name in model_names):
        raise ValueError("Expected equal ensemble weights")
    predictions = {}
    with torch.inference_mode():
        for name in model_names:
            model = build_model(config, node_input_dim, bundle["models"][name]["state_dict"], device)
            amp = nullcontext()
            if config.inference.use_amp and device.type == "cuda":
                dtype = torch.bfloat16 if config.inference.amp_dtype == "bfloat16" else torch.float16
                amp = torch.autocast("cuda", dtype=dtype)
            with amp:
                raw = float(model(graph1, graph2).float().cpu())
            delta_g = raw * std + mean
            if not math.isfinite(delta_g):
                raise ValueError(f"Non-finite prediction from {name}")
            predictions[name] = {
                "raw_output": raw, "delta_g_kcal_mol": delta_g,
            }
            print(f"{name}: {delta_g:.6f} kcal/mol", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ensemble = sum(value["delta_g_kcal_mol"] for value in predictions.values()) / len(model_names)
    return ensemble, predictions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--receptor-chains", nargs="+", required=True)
    parser.add_argument("--ligand-chains", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for features, graphs and prediction.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/model.ckpt", help=argparse.SUPPRESS)
    parser.add_argument("--config-snapshot", type=Path, help="Optional config override; defaults to checkpoint config or configs/config.json")
    parser.add_argument("--esm-model", type=Path, default=Path(torch.hub.get_dir()) / "checkpoints/esm2_t33_650M_UR50D.pt")
    parser.add_argument("--mkdssp", type=Path, default=ROOT / "feature_generate/mkdssp")
    parser.add_argument("--device", default="cpu", help="ESM2 and affinity model device: cpu or cuda:N; NUV uses CPU")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for feature generation and inference")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("pdb", "output_dir", "checkpoint", "config_snapshot", "esm_model", "mkdssp"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    regression = args.esm_model.with_name(args.esm_model.stem + "-contact-regression.pt")
    for path in [args.pdb, args.checkpoint, args.esm_model, regression, args.mkdssp, *([args.config_snapshot] if args.config_snapshot else [])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    if importlib.util.find_spec("esm") is None:
        raise ModuleNotFoundError("Install fair-esm in this Python environment: python -m pip install fair-esm==2.0.1")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bundle, config, node_input_dim, label_stats, config_origin = load_checkpoint(args.checkpoint, args.config_snapshot)
    if not label_stats:
        raise ValueError("Config snapshot must contain training label_stats")
    features = config.features
    if node_input_dim != 1289:
        raise ValueError("Expected refund configuration: ESM2 layer 33 + DSSP, 1289 input channels")
    torch.set_num_threads(args.threads)
    feature_root = args.output_dir / "features"
    command = [
        sys.executable, str(ROOT / "feature_generate/feature_generate.py"),
        "--pdb", str(args.pdb),
        "--receptor-chains", *args.receptor_chains,
        "--ligand-chains", *args.ligand_chains,
        "--output-dir", str(feature_root),
        "--esm-model", str(args.esm_model), "--mkdssp", str(args.mkdssp),
        "--device", str(device), "--knn", str(features.top_k),
        "--max-residues", str(features.max_residues),
    ]
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    print("Generating features and graphs...", flush=True)
    subprocess.run(command, check=True, env=environment)
    metadata = json.loads((feature_root / "metadata.json").read_text())
    graph_path = feature_root / metadata["graph_file"]
    graphs, _ = dgl.load_graphs(str(graph_path))
    if len(graphs) != 2 or any(graph.ndata["f"].shape[1] != node_input_dim for graph in graphs):
        raise ValueError("Expected a receptor/ligand graph pair with 1289 node features")
    is_ensemble = len(bundle["models"]) > 1
    print("Running five-model ensemble..." if is_ensemble else "Running single-model inference...", flush=True)
    ensemble, predictions = predict_ensemble(graphs, config, node_input_dim, label_stats, bundle, device)
    result = {
        "pdb": str(args.pdb),
        "receptor_chains": metadata["receptor_chains"], "ligand_chains": metadata["ligand_chains"],
        "delta_g_kcal_mol": ensemble,
        "ensemble_method": "equal-weight mean of five denormalized Delta G predictions" if is_ensemble else "single-model denormalized Delta G prediction",
        "models": predictions, "training_label_stats": label_stats,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_snapshot": config_origin,
        "device": str(device), "nuv_device": "cpu",
        "feature_metadata": str(feature_root / "metadata.json"), "graph_file": str(graph_path),
    }
    output = args.output_dir / "prediction.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Predicted affinity (Delta G): {ensemble:.6f} kcal/mol", flush=True)
    print(f"Saved prediction: {output}", flush=True)


if __name__ == "__main__":
    main()
