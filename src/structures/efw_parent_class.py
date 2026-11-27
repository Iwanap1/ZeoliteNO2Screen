
from typing import Dict, Tuple, List
from pathlib import Path
import contextlib
import io
import json
import yaml

import numpy as np
import pandas as pd
import spglib
import zse

from ase import Atoms
from ase.geometry import find_mic
from ase.io import write
from zse.cif_tools import read_cif
from zse.rings import get_rings


class ExtraFrameworkStructureGenerator:
    """Base class for generating extraframework species in zeolites."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.framework = None
        self.framework_code = None
        self._neighbors = None
        self._rings = None
        self._symmetry_maps = None

        self.enum = config["enumeration"]
        self.placement = config["placement"]
        self.validation = config["validation"]
        self.output = config["output"]

        self.bond_cutoff = float(self.enum["t_o_bond_cutoff"])
        self.symprec = float(self.enum["symmetry"]["symprec"])
        self.ring_sizes = ()

    def load_framework(self, code: str) -> Atoms:
        """Load an all-silica zeolite framework using ZSE."""
        path = Path(self.config["frameworks"]["directory"]) / f"{code}.cif"

        if code == "BEA" and not path.exists():
            path = path.with_name("Beta_A.cif")

        if not path.exists():
            raise FileNotFoundError(path)

        (Path(zse.__file__).parent / ".temp_files").mkdir(parents=True, exist_ok=True)

        with contextlib.redirect_stdout(io.StringIO()):
            atoms = read_cif(str(path))[0]

        atoms.pbc = True

        if any(a.symbol not in ("Si", "O") for a in atoms):
            raise ValueError(f"{code}: expected an all-silica Si/O framework")

        self.framework = atoms
        self.framework_code = code
        self._neighbors = None
        self._rings = None
        self._symmetry_maps = None

        return atoms

    @property
    def t_indices(self) -> Tuple[int, ...]:
        """Indices of framework tetrahedral sites."""
        return tuple(a.index for a in self.framework if a.symbol == "Si")

    @property
    def t_o_neighbors(self) -> Dict[int, set]:
        """Framework oxygen neighbours of each T atom."""
        if self._neighbors is None:
            o_ids = np.array([a.index for a in self.framework if a.symbol == "O"])
            self._neighbors = {}

            for i in self.t_indices:
                distances = self.framework.get_distances(i, o_ids, mic=True)
                self._neighbors[i] = set(map(int, o_ids[distances < self.bond_cutoff]))

        return self._neighbors

    @property
    def o_t_neighbors(self) -> Dict[int, List[int]]:
        """Framework T neighbours of each oxygen atom."""
        result = {}

        for t, oxygens in self.t_o_neighbors.items():
            for o in oxygens:
                result.setdefault(o, []).append(t)

        return result

    def substitute_al(self, indices: Tuple[int, ...]) -> Atoms:
        """Replace selected framework Si atoms with Al."""
        atoms = self.framework.copy()

        for i in indices:
            if atoms[int(i)].symbol != "Si":
                raise ValueError(f"Atom {i} is not Si")
            atoms[int(i)].symbol = "Al"

        return atoms

    def obeys_lowenstein(self, pair: Tuple[int, int]) -> bool:
        """Check that two Al substitutions do not share a framework oxygen."""
        i, j = map(int, pair)

        return i != j and not (
            self.t_o_neighbors[i] & self.t_o_neighbors[j]
        )

    @property
    def symmetry_maps(self) -> List[Dict[int, int]]:
        """Get T-site permutations under parent-framework symmetry operations."""
        if self._symmetry_maps is None:
            atoms = self.framework
            scaled = atoms.get_scaled_positions(wrap=True)
            cell = np.asarray(atoms.cell)

            symmetry = spglib.get_symmetry(
                (cell, scaled, atoms.get_atomic_numbers()),
                symprec=self.symprec
            )

            if symmetry is None:
                raise RuntimeError("Framework symmetry detection failed")

            ids = np.array(self.t_indices)
            positions = atoms.positions[ids]
            maps = []

            for rotation, translation in zip(
                symmetry["rotations"], symmetry["translations"]
            ):
                transformed = (scaled[ids] @ rotation.T + translation) @ cell
                mapping = {}

                for i, pos in zip(ids, transformed):
                    _, distances = find_mic(positions - pos, atoms.cell, atoms.pbc)
                    j = int(np.argmin(distances))

                    if distances[j] > 0.1:
                        raise RuntimeError("Symmetry mapping failed")

                    mapping[int(i)] = int(ids[j])

                if len(set(mapping.values())) != len(ids):
                    raise RuntimeError("Non-bijective symmetry mapping")

                maps.append(mapping)

            self._symmetry_maps = maps

        return self._symmetry_maps

    def unique_pairs(self, pairs: List[Tuple[int, int]]) -> List[Dict]:
        """Reduce Al-pair configurations using parent-framework symmetry."""
        eligible = {tuple(sorted(map(int, pair))) for pair in pairs}
        remaining = eligible.copy()
        result = []

        while remaining:
            pair = min(remaining)

            equivalents = {
                tuple(sorted((mapping[pair[0]], mapping[pair[1]])))
                for mapping in self.symmetry_maps
            }

            matches = equivalents & eligible

            result.append({
                "pair": pair,
                "multiplicity": len(matches)
            })

            remaining.difference_update(matches)

        return result

    def _rings_around_t(self, t: int) -> List[Dict]:
        """Find rings containing a T atom using ZSE."""
        with contextlib.redirect_stdout(io.StringIO()):
            sizes, paths, _, repeated = get_rings(
                self.framework,
                index=int(t),
                max_ring=max(self.ring_sizes),
                validation=self.enum["ring_validation"]
            )

        fw = self.framework
        n = len(fw)
        symbols = np.asarray(fw.get_chemical_symbols())
        repeated_symbols = np.asarray(repeated.get_chemical_symbols())
        rings = []

        for size, path in zip(sizes, paths):
            size = int(size)

            if size not in self.ring_sizes:
                continue

            path = np.asarray(path, dtype=int)

            if len(path) != 2 * size or path.min() < 0 or path.max() >= len(repeated):
                continue

            ids = path % n

            if not np.array_equal(repeated_symbols[path], symbols[ids]):
                continue

            # Validate the mapping between the repeated and original framework
            _, mapping_dist = find_mic(
                repeated.positions[path] - fw.positions[ids],
                fw.cell, fw.pbc
            )

            if np.max(mapping_dist) > 0.2:
                continue

            is_t = symbols[ids] == "Si"

            if is_t.sum() != size or np.any(is_t == np.roll(is_t, 1)):
                continue

            if len(set(ids[is_t])) != size or len(set(ids[~is_t])) != size:
                continue

            # Reconstruct the complete periodic T-O-T path
            coords = [fw.positions[ids[0]].copy()]
            valid = True

            for a, b in zip(ids[:-1], ids[1:]):
                vector = fw.get_distance(int(a), int(b), mic=True, vector=True)

                if np.linalg.norm(vector) > self.bond_cutoff:
                    valid = False
                    break

                coords.append(coords[-1] + vector)

            if not valid:
                continue

            coords = np.asarray(coords)

            closing = fw.get_distance(
                int(ids[-1]), int(ids[0]), mic=True, vector=True
            )

            if np.linalg.norm(coords[-1] + closing - coords[0]) > 0.2:
                continue

            o_coords = coords[~is_t]
            centre = o_coords.mean(axis=0)

            _, _, vh = np.linalg.svd(o_coords - centre)
            normal = vh[-1] / np.linalg.norm(vh[-1])

            rings.append({
                "size": size,
                "t_indices": tuple(map(int, ids[is_t])),
                "o_indices": tuple(map(int, ids[~is_t])),
                "center": centre,
                "normal": normal,
                "o_coords": o_coords
            })

        return rings

    @property
    def rings(self) -> List[Dict]:
        """Cached catalogue of distinct framework rings."""
        if self._rings is None:
            found = {}

            for t in self.t_indices:
                for ring in self._rings_around_t(t):
                    key = (
                        ring["size"],
                        tuple(sorted(ring["t_indices"])),
                        tuple(sorted(ring["o_indices"]))
                    )

                    found.setdefault(key, ring)

            self._rings = list(found.values())

        return self._rings

    def shared_rings(self, pair: Tuple[int, int]) -> List[Dict]:
        """Find framework rings containing both Al substitution sites."""
        pair_set = set(pair)

        return [
            ring for ring in self.rings
            if pair_set.issubset(ring["t_indices"])
        ]

    def classify_6mr_pair(self, pair: Tuple[int, int]) -> List[str]:
        """Classify two-Al arrangements in shared six-membered rings."""
        arrangements = set()

        for ring in self.shared_rings(pair):
            if ring["size"] != 6:
                continue

            indices = ring["t_indices"]
            a, b = [indices.index(i) for i in pair]
            separation = min(abs(a-b), 6-abs(a-b))

            arrangements.add({
                2: "meta",
                3: "para"
            }.get(separation, "other"))

        return sorted(arrangements)

    def metal_geometry(self, atoms: Atoms, metal_index: int, params: Dict) -> Dict:
        """Calculate framework coordination and steric-clash diagnostics."""
        o_ids = np.array([a.index for a in atoms if a.symbol == "O"])
        t_ids = np.array([a.index for a in atoms if a.symbol in ("Si", "Al")])

        d_o = atoms.get_distances(metal_index, o_ids, mic=True)
        d_t = atoms.get_distances(metal_index, t_ids, mic=True)

        coordinating = []
        o_to_t = self.o_t_neighbors

        for o, distance in zip(o_ids, d_o):
            if distance > params["coord_cutoff"]:
                continue

            t_neighbors = o_to_t.get(int(o), [])
            bond_type = "-O-".join(sorted(atoms[i].symbol for i in t_neighbors))

            coordinating.append({
                "o_index": int(o),
                "distance_A": round(float(distance), 4),
                "t_indices": list(map(int, t_neighbors)),
                "bond_type": bond_type,
                "al_adjacent": any(atoms[i].symbol == "Al" for i in t_neighbors)
            })

        coordinating.sort(key=lambda x: x["distance_A"])

        min_mo = float(d_o.min())
        min_mt = float(d_t.min())

        o_clash = min_mo < params["min_mo"] - 1e-3
        t_clash = min_mt < params["min_mt"] - 1e-3

        return {
            "coordination_number": len(coordinating),
            "min_mo_A": min_mo,
            "min_mt_A": min_mt,
            "metal_o_clash": bool(o_clash),
            "metal_t_clash": bool(t_clash),
            "steric_clash": bool(o_clash or t_clash),
            "low_coordination": len(coordinating) < self.validation["minimum_coordination"],
            "n_al_adjacent_o": sum(o["al_adjacent"] for o in coordinating),
            "n_si_o_si": sum(o["bond_type"] == "Si-O-Si" for o in coordinating),
            "coordinating_oxygens": coordinating
        }

    @staticmethod
    def summary(sites: List[Atoms]) -> pd.DataFrame:
        """Create a CSV-ready DataFrame from ASE structure metadata."""
        rows = []

        for atoms in sites:
            metadata = atoms.info["metadata"]
            row = {**metadata, **metadata.get("geometry", {})}
            row.pop("geometry", None)

            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    row[key] = json.dumps(value, default=str)
                elif isinstance(value, np.generic):
                    row[key] = value.item()

            rows.append(row)

        return pd.DataFrame(rows)

    def export(self, sites: List[Atoms]) -> pd.DataFrame:
        """Export generated structures, diagnostics and configuration."""
        root = Path(self.output["directory"])
        root.mkdir(parents=True, exist_ok=True)

        df = self.summary(sites)

        if self.output.get("save_diagnostics_csv", True):
            df.to_csv(root / "diagnostics.csv", index=False)

        if self.output.get("save_structures", True):
            fmt = self.output.get("structure_format", "extxyz")
            folder = root / "structures"
            folder.mkdir(exist_ok=True)

            for atoms in sites:
                metadata = atoms.info["metadata"]
                site_id = metadata["site_id"]

                # Keep nested metadata in the diagnostics, not the extxyz header
                structure = atoms.copy()
                structure.info = {
                    "site_id": site_id,
                    "framework": metadata["framework"],
                    "cation": metadata["cation"],
                    "efw_site_type": metadata["efw_site_type"]
                }

                write(folder / f"{site_id}.{fmt}", structure, format=fmt)

        with open(root / "generation_config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        if self.output.get("save_metadata_json", False):
            with open(root / "metadata.json", "w") as f:
                json.dump(
                    [atoms.info["metadata"] for atoms in sites],
                    f,
                    indent=2,
                    default=lambda x: (
                        x.tolist() if isinstance(x, np.ndarray)
                        else x.item() if isinstance(x, np.generic)
                        else str(x)
                    )
                )

        return df
