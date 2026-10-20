"""Inference-only config loading and affinity model with refund checkpoint keys."""

import json
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
from e3nn.nn import BatchNorm as E3BatchNorm

from trigeo_encoder import TriGeo


def load_snapshot(path):
    if isinstance(path, dict):
        payload = path
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    config = SimpleNamespace(**{
        section: SimpleNamespace(**payload[section])
        for section in ("features", "model")
    })
    config.inference = SimpleNamespace(**payload.get("inference", payload.get("train", {})))
    return config, int(payload["node_input_dim"]), payload.get("label_stats")


def _graph_instance_forward(self, input_tensor):
    if input_tensor.ndim == 2:
        return E3BatchNorm.forward(self, input_tensor.unsqueeze(0)).squeeze(0)
    return E3BatchNorm.forward(self, input_tensor)


def enable_graph_instance_norm(module):
    for child in module.modules():
        if isinstance(child, E3BatchNorm):
            child.instance = True
            child.forward = types.MethodType(_graph_instance_forward, child)


class DualPool(nn.Module):
    """Pool both partners and their pair embedding into one affinity value."""

    def __init__(self, dim, hidden, dropout, use_z, interface_pool, interface_tau):
        super().__init__()
        self.use_z = use_z
        self.interface_pool = interface_pool
        self.interface_tau = interface_tau
        self.norm = nn.LayerNorm(dim)
        if use_z:
            self.z_score = nn.Linear(dim, 1)
            self.z_norm = nn.LayerNorm(dim)
        input_dim = 3 * dim + (dim if use_z else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, receptor_features, ligand_features, pairs, receptor_ca, ligand_ca):
        if self.interface_pool:
            distances = torch.cdist(receptor_ca.float(), ligand_ca.float())
            receptor_weights = torch.softmax(-distances.min(dim=1).values / self.interface_tau, dim=0)
            ligand_weights = torch.softmax(-distances.min(dim=0).values / self.interface_tau, dim=0)
            receptor = self.norm((receptor_weights.unsqueeze(-1) * receptor_features).sum(dim=0))
            ligand = self.norm((ligand_weights.unsqueeze(-1) * ligand_features).sum(dim=0))
        else:
            receptor = self.norm(receptor_features.mean(dim=0))
            ligand = self.norm(ligand_features.mean(dim=0))
        features = [receptor + ligand, (receptor - ligand).abs(), receptor * ligand]
        if self.use_z:
            flattened = pairs.reshape(-1, pairs.shape[-1])
            weights = torch.softmax(self.z_score(flattened), dim=0)
            features.append(self.z_norm((weights * flattened).sum(dim=0)))
        return self.mlp(torch.cat(features, dim=-1))


class TriGeoAffinity(nn.Module):
    """Evaluation model using trigeo.* and pool.* state keys."""

    def __init__(self, node_input_dim, config):
        super().__init__()
        if config.readout != "dual":
            raise ValueError("This inference model requires a DualPool checkpoint (readout=dual)")
        hidden = config.hidden_dim
        self.trigeo = TriGeo(
            num_node_input_feats=hidden,
            num_edge_input_feats=28,
            num_classes=2,
            num_gnn_hidden_channels=hidden,
            output_emb=True,
            num_interact_layers=config.num_interact_layers,
        )
        self.trigeo.other_embedding = False
        if config.node_stem == "mlp":
            self.trigeo.node_in_embedding = nn.Sequential(
                nn.LayerNorm(node_input_dim), nn.Linear(node_input_dim, hidden),
                nn.GELU(), nn.LayerNorm(hidden),
            )
        elif config.node_stem == "linear":
            self.trigeo.node_in_embedding = nn.Linear(node_input_dim, hidden, bias=False)
        else:
            raise ValueError(f"Unknown node_stem: {config.node_stem}")
        if hasattr(self.trigeo, "node_other_in_embedding"):
            del self.trigeo.node_other_in_embedding
        if config.i3nn_instance_norm:
            enable_graph_instance_norm(self.trigeo)
        self.pool = DualPool(
            dim=hidden, hidden=max(config.pool_hidden, hidden), dropout=0.0,
            use_z=config.dual_use_z, interface_pool=config.interface_pool,
            interface_tau=config.interface_tau,
        )

    def forward(self, graph1, graph2):
        with graph1.local_scope(), graph2.local_scope():
            receptor_ca, ligand_ca = graph1.ndata["x"], graph2.ndata["x"]
            output = self.trigeo(graph1, graph2)
            return self.pool(output[1], output[2], output[3], receptor_ca, ligand_ca)


def build_model(config, node_input_dim, checkpoint, device):
    model = TriGeoAffinity(node_input_dim, config.model)
    state_dict = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
