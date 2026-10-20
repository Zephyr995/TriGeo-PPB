r"""Generate refund-compatible features and a receptor/ligand DGL graph pair.

Example:
    python feature_generate.py --pdb complex.pdb --receptor-chains A B \
        --ligand-chains C --esm-model /models/esm2_t33_650M_UR50D.pt \
        --output-dir features/complex

The local ESM2 checkpoint requires its sibling *-contact-regression.pt file.
Outputs: pdb_extract/*.pdb, sequences.fa, pdb/*.tensor, DSSP/*.tensor,
NUV/*.tensor, esm2/*.pt, graphs/*.bin and metadata.json. ESM2 uses the first 1022 residues and zero-pads the remainder for graph input,
matching the existing extraction and FeatureStore pipeline. XYZ, DSSP and NUV
remain full length.
"""

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as functional
from Bio import PDB, pairwise2
from dgl.nn.pytorch import pairwise_squared_distance


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T",
    "PTR": "Y", "HYP": "P", "CSO": "C", "CSD": "C", "OCS": "C",
    "KCX": "K", "MLY": "K", "M3L": "K", "LLP": "K", "CME": "C",
    "CSS": "C", "CSX": "C", "FME": "M", "HIC": "H", "HSD": "H",
    "HSE": "H", "HSP": "H",
}
ESM_LAYER = 33
ESM_LENGTH = 1022


def parse_chains(values):
    chains = list(dict.fromkeys(
        character for value in values for character in value
        if character not in ", "
    ))
    if not chains:
        raise ValueError("At least one chain ID is required for each partner")
    return chains


def split_pdb(pdb_path, receptor_chains, ligand_chains, output_paths):
    """Keep the first model and preserve chain and residue order within each side."""
    model = next(PDB.PDBParser(QUIET=True).get_structure("complex", str(pdb_path)).get_models())
    present = {chain.id for chain in model}
    missing = set(receptor_chains + ligand_chains) - present
    if missing:
        raise ValueError(f"Missing chains {sorted(missing)}; available: {sorted(present)}")
    if set(receptor_chains) & set(ligand_chains):
        raise ValueError("Receptor and ligand chain IDs must not overlap")

    class ChainSelect(PDB.Select):
        def __init__(self, chain_ids):
            self.chain_ids = set(chain_ids)

        def accept_chain(self, chain):
            return chain.id in self.chain_ids

    writer = PDB.PDBIO()
    writer.set_structure(model)
    for chains, output in zip((receptor_chains, ligand_chains), output_paths):
        writer.save(str(output), ChainSelect(chains))


def sequence_and_xyz(pdb_path):
    """Select residues once for both sequence and [N, CA, C, O, sidechain] XYZ."""
    sequence = []
    coordinates = []
    atoms = {}
    residue_key = None
    residue_name = None
    model_count = 0

    def flush():
        if not atoms:
            return
        zero = np.zeros(3, dtype=np.float32)
        backbone = [atoms.get(atom, zero) for atom in ("N", "CA", "C", "O")]
        sidechain = [coord for atom, coord in atoms.items() if atom not in ("N", "CA", "C", "O")]
        center = np.mean(sidechain, axis=0) if sidechain else atoms.get("CA", zero)
        coordinates.append(np.stack(backbone + [center]))
        sequence.append(THREE_TO_ONE.get(residue_name, "X"))

    for line in pdb_path.read_text().splitlines():
        tag = line[:6].strip()
        if tag == "MODEL":
            model_count += 1
            if model_count > 1:
                break
        elif tag in ("ENDMDL", "TER"):
            flush()
            atoms = {}
            residue_key = None
            if tag == "ENDMDL":
                break
        elif tag in ("ATOM", "HETATM") and len(line) >= 54:
            name = line[17:20].strip()
            if tag == "HETATM" and name not in THREE_TO_ONE:
                continue
            try:
                key = (line[21], int(line[22:26]), line[26])
            except ValueError:
                continue
            if key != residue_key:
                flush()
                atoms = {}
                residue_key, residue_name = key, name
            atom = line[12:16].strip()
            if line[16] not in (" ", "A") or not atom or atom.startswith("H"):
                continue
            try:
                atoms[atom] = np.array([line[30:38], line[38:46], line[46:54]], dtype=np.float32)
            except ValueError:
                continue
    flush()
    if not coordinates:
        raise ValueError(f"No protein residues found in {pdb_path}")
    xyz = torch.from_numpy(np.stack(coordinates)).float()
    if not torch.isfinite(xyz).all():
        raise ValueError(f"Non-finite coordinates in {pdb_path}")
    return "".join(sequence), xyz


def parse_dssp(path, reference_sequence):
    """Use the original GPSite RSA normalization and HBEGITSC one-hot order."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    secondary_structures = "HBEGITSC"
    standard_asa = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]
    sequence = []
    features = []
    header_found = False
    for line in path.read_text().splitlines():
        if not header_found:
            header_found = line.lstrip().startswith("#")
            continue
        if len(line) < 38 or line[13] in ("!", "*"):
            continue
        amino_acid = line[13]
        secondary_structure = line[16] if line[16] != " " else "C"
        feature = np.zeros(9, dtype=np.float32)
        feature[0] = min(1, float(line[34:38]) / standard_asa[amino_acids.find(amino_acid)])
        feature[1 + (secondary_structures.find(secondary_structure) % 8)] = 1
        sequence.append(amino_acid)
        features.append(feature)
    if not features:
        raise ValueError(f"No DSSP records found in {path}")
    sequence = "".join(sequence)
    if sequence != reference_sequence:
        alignment = pairwise2.align.globalxx(reference_sequence, sequence, one_alignment_only=True)[0]
        matched = []
        feature_index = 0
        for reference, observed in zip(alignment.seqA, alignment.seqB):
            feature = np.zeros(9, dtype=np.float32) if observed == "-" else features[feature_index]
            if observed != "-":
                feature_index += 1
            if reference != "-":
                matched.append(feature)
        features = matched
    return torch.from_numpy(np.stack(features)).float()


def compute_dssp(pdb_path, sequence, executable):
    """Retry DSSP with HETATM removed and with a Biopython rewrite."""
    errors = []
    with tempfile.TemporaryDirectory(prefix="tri_geo_dssp_") as temporary:
        directory = Path(temporary)
        output = directory / "output.dssp"
        candidates = [pdb_path, directory / "nohet.pdb", directory / "rewritten.pdb"]
        candidates[1].write_text("".join(
            line for line in pdb_path.read_text().splitlines(keepends=True)
            if not line.startswith("HETATM")
        ))

        class NoHeteroResidues(PDB.Select):
            def accept_residue(self, residue):
                return residue.id[0] == " "

        writer = PDB.PDBIO()
        writer.set_structure(PDB.PDBParser(QUIET=True).get_structure("protein", str(pdb_path)))
        writer.save(str(candidates[2]), NoHeteroResidues())
        for candidate in candidates:
            output.unlink(missing_ok=True)
            try:
                result = subprocess.run(
                    [str(executable), "-i", str(candidate), "-o", str(output)],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0 and output.is_file():
                    return parse_dssp(output, sequence)
                errors.append(f"{candidate.name}: {result.stderr.strip()}")
            except (subprocess.TimeoutExpired, ValueError) as error:
                errors.append(f"{candidate.name}: {error}")
    raise RuntimeError(f"DSSP failed for {pdb_path}: {'; '.join(errors)}")


def compute_nuv(xyz, device="cpu"):
    """Compute the ProTact soft-distance gradient normal and Pixar tangents."""
    from pykeops.torch import LazyTensor

    ca = xyz[:, 1, :].contiguous().to(device)
    valid = ca.abs().sum(dim=-1) > 0
    frames = torch.zeros(xyz.shape[0], 3, 3, dtype=torch.float32)
    if not valid.any():
        return frames
    centers = ca[valid].contiguous()
    with torch.enable_grad():
        samples = centers.detach().clone().requires_grad_(True)
        center_tensor = LazyTensor(centers[:, None, :])
        sample_tensor = LazyTensor(samples[None, :, :])
        squared_distances = ((center_tensor - sample_tensor) ** 2).sum(-1)
        distances = -((-squared_distances.sqrt()).logsumexp(dim=0)).reshape(-1)
        gradient = torch.autograd.grad(distances.sum(), samples)[0]
        normals = functional.normalize(gradient, p=2, dim=-1)
    normal_x, normal_y, normal_z = normals.unbind(-1)
    sign = 2 * (normal_z >= 0) - 1.0
    scale = -1 / (sign + normal_z)
    product = normal_x * normal_y * scale
    tangents = torch.stack((
        1 + sign * normal_x * normal_x * scale, sign * product, -sign * normal_x,
        product, sign + normal_y * normal_y * scale, -normal_y,
    ), dim=-1).reshape(-1, 2, 3)
    frames[valid.cpu()] = torch.cat((normals[:, None, :], tangents), dim=1).detach().cpu()
    if not torch.isfinite(frames).all():
        raise ValueError("NUV computation produced non-finite values")
    return frames


def compute_esm2(sequences, checkpoint, output_directory, device):
    """Load one local ESM2 model and extract both partners sequentially."""
    from esm import pretrained

    model, alphabet = pretrained.load_model_and_alphabet_local(str(checkpoint))
    if model.num_layers != ESM_LAYER or model.embed_dim != 1280:
        raise ValueError("Expected esm2_t33_650M_UR50D (33 layers, 1280 channels)")
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter(ESM_LENGTH)
    embeddings = {}
    with torch.no_grad():
        for name, sequence in sequences.items():
            if len(sequence) > ESM_LENGTH:
                warnings.warn(f"{name}: ESM2 truncates {len(sequence)} to {ESM_LENGTH}; graph input tail is zero-padded")
            _, _, tokens = converter([(name, sequence)])
            output = model(tokens.to(device), repr_layers=[ESM_LAYER], return_contacts=False)
            embedding = output["representations"][ESM_LAYER][0, 1:min(len(sequence), ESM_LENGTH) + 1].cpu().clone()
            torch.save({"label": name, "representations": {ESM_LAYER: embedding}}, output_directory / f"{name}.pt")
            embeddings[name] = functional.pad(embedding, (0, 0, 0, len(sequence) - embedding.shape[0]))
            print(f"ESM2: {name}, {tuple(embedding.shape)}", flush=True)
    return embeddings


def gather_neighbors(values, indices):
    flattened = indices.reshape(indices.shape[0], -1).unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, 1, flattened).reshape(*indices.shape, values.shape[-1])


def geometric_edge_features(ca, neighbor_indices, squared_distances):
    """Match the original geometric RBF, local direction and quaternion features."""
    coordinates = ca.unsqueeze(0)
    unit_bonds = functional.normalize(coordinates[:, 1:] - coordinates[:, :-1], dim=-1)
    previous, current = unit_bonds[:, :-2], unit_bonds[:, 1:-1]
    normals = functional.normalize(torch.cross(previous, current), dim=-1)
    first_axis = functional.normalize(previous - current, dim=-1)
    frames = torch.stack((first_axis, normals, torch.cross(first_axis, normals)), dim=2)
    frames = functional.pad(frames.reshape(1, -1, 9), (0, 0, 1, 2))
    if ca.shape[0] < 3:
        frames = torch.zeros(1, ca.shape[0], 9)
    neighbor_frames = gather_neighbors(frames, neighbor_indices).reshape(1, ca.shape[0], -1, 3, 3)
    frames = frames.reshape(1, ca.shape[0], 3, 3)
    displacement = gather_neighbors(coordinates, neighbor_indices) - coordinates.unsqueeze(2)
    directions = functional.normalize(torch.matmul(frames.unsqueeze(2), displacement.unsqueeze(-1)).squeeze(-1), dim=-1)
    rotations = torch.matmul(frames.unsqueeze(2).transpose(-1, -2), neighbor_frames)
    diagonal = torch.diagonal(rotations, dim1=-2, dim2=-1)
    diagonal_x, diagonal_y, diagonal_z = diagonal.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack((
        diagonal_x - diagonal_y - diagonal_z,
        -diagonal_x + diagonal_y - diagonal_z,
        -diagonal_x - diagonal_y + diagonal_z,
    ), dim=-1)))
    signs = torch.sign(torch.stack((
        rotations[..., 2, 1] - rotations[..., 1, 2],
        rotations[..., 0, 2] - rotations[..., 2, 0],
        rotations[..., 1, 0] - rotations[..., 0, 1],
    ), dim=-1))
    scalar = torch.sqrt(functional.relu(1 + diagonal.sum(-1, keepdim=True))) / 2
    quaternions = functional.normalize(torch.cat((signs * magnitudes, scalar), dim=-1), dim=-1)
    centers = torch.linspace(0, 20, 18).reshape(1, 1, 1, -1)
    radial = torch.exp(-((squared_distances.unsqueeze(-1) - centers) / (20 / 18)) ** 2)
    return torch.cat((radial, directions, quaternions), dim=-1).reshape(-1, 25)


def build_graph(xyz, node_features, nuv, knn=20):
    """Use KNN source indices as neighbors, as in build_protact_graph_fixed."""
    length = xyz.shape[0]
    ca = xyz[:, 1, :].contiguous().float()
    neighbors = min(knn, length)
    graph = dgl.knn_graph(ca, neighbors)
    source, destination = graph.edges()
    squared_distances = torch.topk(pairwise_squared_distance(ca), neighbors, 1, largest=False).values
    geometry = geometric_edge_features(ca, source.reshape(1, length, neighbors), squared_distances)
    distances = ((ca[source] - ca[destination]) ** 2).sum(dim=1)
    distance_range = distances.max() - distances.min()
    weights = (distances - distances.min()) / distance_range if distance_range > 0 else torch.zeros_like(distances)
    graph.ndata["f"] = node_features.float()
    graph.ndata["x"] = ca
    graph.ndata["nuv"] = nuv.float()
    graph.edata["f"] = torch.nan_to_num(torch.cat((
        torch.sin((source - destination).float()).unsqueeze(1),
        weights.unsqueeze(1), geometry, torch.zeros(source.shape[0], 1),
    ), dim=1))
    return graph


def truncate_pair(receptor, ligand, max_residues):
    """Select each partner against the full-length other partner, then sort indices."""
    selected = []
    for protein, partner in ((receptor, ligand), (ligand, receptor)):
        indices = torch.arange(protein[0].shape[0])
        if max_residues and len(indices) > max_residues:
            distances = torch.cdist(protein[0][:, 1, :].float(), partner[0][:, 1, :].float())
            indices = torch.topk(distances.min(dim=1).values, max_residues, largest=False).indices.sort().values
        selected.append((tuple(value[indices] for value in protein), indices))
    return selected


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--receptor-chains", nargs="+", required=True, help="Single-character PDB chain IDs, e.g. A B or A,B")
    parser.add_argument("--ligand-chains", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new or empty output directory")
    parser.add_argument("--name", help="Output ID; defaults to the lowercase PDB filename stem")
    parser.add_argument("--esm-model", type=Path, required=True, help="Local esm2_t33_650M_UR50D.pt checkpoint")
    parser.add_argument("--mkdssp", type=Path, default=Path(__file__).resolve().with_name("mkdssp"))
    parser.add_argument("--device", default="cpu", help="ESM2 device: cpu or cuda:N; NUV always uses CPU")
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--max-residues", type=int, default=400, help="Interface truncation per partner; 0 disables it")
    return parser


def main():
    args = build_parser().parse_args()
    args.pdb = args.pdb.resolve()
    args.esm_model = args.esm_model.resolve()
    args.mkdssp = args.mkdssp.resolve()
    for path in (args.pdb, args.esm_model, args.mkdssp):
        if not path.is_file():
            raise FileNotFoundError(path)
    regression = args.esm_model.with_name(args.esm_model.stem + "-contact-regression.pt")
    if not regression.is_file():
        raise FileNotFoundError(f"ESM2 companion checkpoint required: {regression}")
    if args.knn < 1 or args.max_residues < 0:
        raise ValueError("knn must be positive and max-residues must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    name = args.name or args.pdb.stem.lower()
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("name must be a filename without directory components")
    receptor_chains = parse_chains(args.receptor_chains)
    ligand_chains = parse_chains(args.ligand_chains)
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {root}")
    for subdirectory in ("pdb_extract", "pdb", "DSSP", "NUV", "esm2", "graphs"):
        (root / subdirectory).mkdir(parents=True, exist_ok=True)
    names = [f"{name}_rec", f"{name}_lig"]
    split_paths = [root / "pdb_extract" / f"{protein_name}.pdb" for protein_name in names]
    split_pdb(args.pdb, receptor_chains, ligand_chains, split_paths)
    sequences = {}
    proteins = {}
    for protein_name, pdb_path in zip(names, split_paths):
        sequence, xyz = sequence_and_xyz(pdb_path)
        sequences[protein_name] = sequence
        torch.save(xyz, root / "pdb" / f"{protein_name}.tensor")
        dssp = compute_dssp(pdb_path, sequence, args.mkdssp)
        torch.save(dssp, root / "DSSP" / f"{protein_name}.tensor")
        nuv = compute_nuv(xyz)
        torch.save(nuv, root / "NUV" / f"{protein_name}.tensor")
        proteins[protein_name] = (xyz, dssp, nuv)
        print(f"XYZ/DSSP/NUV: {protein_name}, {len(sequence)} residues", flush=True)
    (root / "sequences.fa").write_text("".join(f">{protein_name}\n{sequence}\n" for protein_name, sequence in sequences.items()))
    embeddings = compute_esm2(sequences, args.esm_model, root / "esm2", str(device))
    full_proteins = []
    for protein_name in names:
        xyz, dssp, nuv = proteins[protein_name]
        node_features = torch.cat((embeddings[protein_name], dssp), dim=-1)
        if node_features.shape != (len(sequences[protein_name]), 1289):
            raise ValueError(f"Unexpected node feature shape for {protein_name}: {node_features.shape}")
        if not torch.isfinite(node_features).all():
            raise ValueError(f"Non-finite node features for {protein_name}")
        full_proteins.append((xyz, node_features, nuv))
    selected = truncate_pair(*full_proteins, args.max_residues)
    graphs = [build_graph(*protein, knn=args.knn) for protein, _ in selected]
    graph_path = root / "graphs" / f"{names[0]}__{names[1]}.bin"
    dgl.save_graphs(str(graph_path), graphs, {"rec_indices": selected[0][1], "lig_indices": selected[1][1]})
    metadata = {
        "pdb": str(args.pdb), "name": name,
        "receptor_chains": receptor_chains, "ligand_chains": ligand_chains,
        "esm_model": str(args.esm_model), "esm_layer": ESM_LAYER,
        "esm_max_length": ESM_LENGTH, "device": str(device), "nuv_device": "cpu",
        "knn": args.knn, "max_residues": args.max_residues,
        "graph_builder": "refund_knn_src_fixed", "graph_order": names,
        "graph_file": str(graph_path.relative_to(root)),
        "residues": {protein_name: len(sequences[protein_name]) for protein_name in names},
        "graph_nodes": {protein_name: graph.num_nodes() for protein_name, graph in zip(names, graphs)},
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved receptor/ligand graphs: {graph_path}", flush=True)


if __name__ == "__main__":
    main()
