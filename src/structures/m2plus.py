
from typing import Dict, Tuple, List
from pathlib import Path
from itertools import combinations, product

import numpy as np
from ase import Atom, Atoms
from ase.geometry import find_mic
from scipy.optimize import minimize

from .efw_parent_class import ExtraFrameworkStructureGenerator


class M2PlusGenerator(ExtraFrameworkStructureGenerator):
    """Generate charge-balanced Z2M2+ zeolite structures from YAML configuration."""

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.species = self.config["species"]["M2+"]
        self.ring_sizes = tuple(self.species["ring_sizes"])

    def candidate_pairs(self) -> List[Dict]:
        """Find symmetry-distinct Al pairs sharing an allowed framework ring."""
        pairs = set()
        max_distance = self.enum.get("max_al_distance")

        for ring in self.rings:
            for pair in combinations(sorted(ring["t_indices"]), 2):
                pair = tuple(map(int, pair))

                if self.enum.get("enforce_lowenstein", True) and not self.obeys_lowenstein(pair):
                    continue

                if max_distance is not None:
                    if self.framework.get_distance(*pair, mic=True) > float(max_distance):
                        continue

                pairs.add(pair)

        if not self.enum.get("symmetry", {}).get("enabled", True):
            return [{"pair": pair, "multiplicity": 1} for pair in sorted(pairs)]

        return self.unique_pairs(pairs)

    def _local_environment(self, centre: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct periodic framework images around the selected ring."""
        fw = self.framework
        cell = np.asarray(fw.cell)
        radius = float(self.placement["search_radius"]) + 4.0

        vectors, _ = find_mic(fw.positions - centre, fw.cell, fw.pbc)

        positions = np.concatenate([
            vectors + i*cell[0] + j*cell[1] + k*cell[2]
            for i, j, k in product((-1, 0, 1), repeat=3)
        ])
        indices = np.tile(np.arange(len(fw)), 27)

        mask = np.linalg.norm(positions, axis=1) < radius
        positions = positions[mask]
        indices = indices[mask]

        symbols = np.asarray(fw.get_chemical_symbols())[indices]
        o_mask = symbols == "O"
        t_mask = np.isin(symbols, ("Si", "Al"))

        return positions[o_mask], indices[o_mask], positions[t_mask]

    def _optimise_position(self, pair: Tuple[int, int], ring: Dict, params: Dict) -> Dict:
        """Optimise metal position using oxygen coordination and steric constraints."""
        centre = np.asarray(ring["center"], dtype=float)
        normal = np.asarray(ring["normal"], dtype=float)
        o_xyz, o_indices, t_xyz = self._local_environment(centre)

        target = float(params["target_mo"])
        min_mo = float(params["min_mo"])
        min_mt = float(params["min_mt"])

        radius = float(self.placement["search_radius"])
        width = float(self.placement.get("oxygen_attraction_width", 0.45))
        al_weight = float(self.placement.get("al_oxygen_preference", 0.35))
        confinement = float(self.placement.get("ring_confinement_weight", 0.03))

        al_oxygens = self.t_o_neighbors[pair[0]] | self.t_o_neighbors[pair[1]]
        al_mask = np.isin(o_indices, list(al_oxygens))

        def distances(x: np.ndarray, xyz: np.ndarray) -> np.ndarray:
            return np.linalg.norm(xyz - x, axis=1)

        def objective(x: np.ndarray) -> float:
            d = distances(x, o_xyz)
            attraction = np.exp(-0.5 * ((d - target) / width)**2)

            return float(
                -np.sum(attraction)
                - al_weight * np.sum(attraction[al_mask])
                + confinement * np.dot(x, x)
            )

        constraints = [
            {"type": "ineq", "fun": lambda x: distances(x, o_xyz).min() - min_mo},
            {"type": "ineq", "fun": lambda x: distances(x, t_xyz).min() - min_mt},
            {"type": "ineq", "fun": lambda x: radius**2 - np.dot(x, x)}
        ]

        seeds = [
            np.zeros(3),
            0.3*normal, -0.3*normal,
            0.8*normal, -0.8*normal
        ]

        groups = [sorted(self.t_o_neighbors[i]) for i in pair]

        for o1 in groups[0]:
            for o2 in groups[1]:
                p1 = self.framework.positions[o1]
                vector = self.framework.get_distance(o1, o2, mic=True, vector=True)
                midpoint = p1 + 0.5 * vector

                delta, _ = find_mic(
                    midpoint - centre, self.framework.cell, self.framework.pbc
                )

                for sign in (-1, 0, 1):
                    seed = delta + sign * 0.3 * normal
                    if np.linalg.norm(seed) <= radius:
                        seeds.append(seed)

        unique_seeds = []

        for seed in sorted(seeds, key=objective):
            if all(np.linalg.norm(seed - old) > 0.2 for old in unique_seeds):
                unique_seeds.append(np.asarray(seed, dtype=float))

            if len(unique_seeds) >= int(self.placement["max_seeds"]):
                break

        opt = self.placement["optimizer"]
        candidates = []

        for seed in unique_seeds:
            result = minimize(
                objective, seed,
                method=opt["method"],
                constraints=constraints,
                bounds=[(-radius, radius)] * 3,
                options={
                    "maxiter": int(opt["maxiter"]),
                    "ftol": float(opt["ftol"])
                }
            )

            x = np.asarray(result.x, dtype=float)
            d_o = distances(x, o_xyz)
            d_t = distances(x, t_xyz)

            violation = (
                max(min_mo - d_o.min(), 0.0)**2
                + max(min_mt - d_t.min(), 0.0)**2
                + max(np.linalg.norm(x) - radius, 0.0)**2
            )

            feasible = (
                d_o.min() >= min_mo - 1e-3
                and d_t.min() >= min_mt - 1e-3
                and np.linalg.norm(x) <= radius + 1e-3
            )

            candidates.append({
                "position": centre + x,
                "score": float(objective(x)),
                "constraint_violation": float(violation),
                "feasible": bool(feasible),
                "converged": bool(result.success),
                "optimizer_message": str(result.message)
            })

        if not candidates:
            raise RuntimeError(f"No placement candidates for Al pair {pair}")

        return min(candidates, key=lambda c: (
            not c["feasible"],
            c["constraint_violation"],
            not c["converged"],
            c["score"]
        ))

    def _make_site(self, pair: Tuple[int, int], ring: Dict, multiplicity: int, metal: str) -> Atoms:
        """Create an M2+ structure with metadata stored in atoms.info."""
        params = self.config["metal_geometry"][metal]
        placement = self._optimise_position(pair, ring, params)

        atoms = self.substitute_al(pair)
        atoms.append(Atom(metal, position=placement["position"]))
        atoms.wrap()

        geometry = self.metal_geometry(atoms, len(atoms)-1, params)
        geometry.update({
            "placement_score": placement["score"],
            "constraint_violation": placement["constraint_violation"],
            "optimizer_converged": placement["converged"],
            "placement_feasible": placement["feasible"],
            "optimizer_message": placement["optimizer_message"]
        })

        geometry["placement_success"] = bool(
            placement["feasible"] and not geometry["steric_clash"]
        )

        if geometry["steric_clash"]:
            geometry["geometry_status"] = "clash"
        elif not placement["feasible"]:
            geometry["geometry_status"] = "placement_failed"
        elif geometry["low_coordination"]:
            geometry["geometry_status"] = "low_coordination"
        else:
            geometry["geometry_status"] = "ok"

        shared = self.shared_rings(pair)

        metadata = {
            "site_id": None,
            "framework": self.framework_code,
            "cation": metal,
            "cation_charge": 2,
            "efw_site_type": "M2+",
            "chemical_formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "n_framework_al": 2,
            "al_indices": list(pair),
            "al_al_distance_A": float(atoms.get_distance(*pair, mic=True)),
            "al_pair_multiplicity": int(multiplicity),
            "lowenstein_valid": bool(self.obeys_lowenstein(pair)),
            "shared_ring_sizes": sorted({int(r["size"]) for r in shared}),
            "n_shared_rings": len(shared),
            "six_mr_arrangements": self.classify_6mr_pair(pair),
            "source_ring_size": int(ring["size"]),
            "source_ring_t_indices": list(map(int, ring["t_indices"])),
            "source_ring_o_indices": list(map(int, ring["o_indices"])),
            "generation_ring_sizes": [int(ring["size"])],
            "placement_method": self.placement["method"],
            "metal_position_cart_A": atoms.positions[-1].tolist(),
            "metal_position_frac": atoms.get_scaled_positions(wrap=True)[-1].tolist(),
            "geometry": geometry
        }

        atoms.info["metadata"] = metadata
        return atoms

    def _site_rank(self, atoms: Atoms) -> Tuple:
        """Rank starting geometries, prioritising feasible, clash-free sites."""
        geometry = atoms.info["metadata"]["geometry"]

        return (
            bool(geometry["steric_clash"]),
            not bool(geometry["placement_feasible"]),
            bool(geometry["low_coordination"]),
            float(geometry["placement_score"])
        )

    def _deduplicate(self, sites: List[Atoms]) -> List[Atoms]:
        """Merge nearly identical positions for the same Al configuration."""
        tolerance = float(self.placement["dedup_tolerance"])
        unique = []

        for atoms in sorted(sites, key=self._site_rank):
            metadata = atoms.info["metadata"]
            duplicate = None

            for existing in unique:
                existing_meta = existing.info["metadata"]

                if metadata["cation"] != existing_meta["cation"]:
                    continue
                if metadata["framework"] != existing_meta["framework"]:
                    continue
                if metadata["al_indices"] != existing_meta["al_indices"]:
                    continue

                _, distance = find_mic(
                    atoms.positions[-1] - existing.positions[-1],
                    atoms.cell, atoms.pbc
                )

                if float(distance) < tolerance:
                    duplicate = existing
                    break

            if duplicate is None:
                unique.append(atoms)
            else:
                existing_meta = duplicate.info["metadata"]
                sizes = set(existing_meta["generation_ring_sizes"])
                sizes.update(metadata["generation_ring_sizes"])
                existing_meta["generation_ring_sizes"] = sorted(sizes)

        return unique

    def _generate_one(self, framework_code: str, metal: str, pairs: List[Dict]) -> List[Atoms]:
        sites = []

        for entry in pairs:
            pair = tuple(entry["pair"])
            multiplicity = int(entry["multiplicity"])
            shared = self.shared_rings(pair)

            for size in self.ring_sizes:
                rings = [r for r in shared if r["size"] == size]
                candidates = [
                    self._make_site(pair, ring, multiplicity, metal)
                    for ring in rings
                ]

                if self.enum.get("one_per_pair_ring", True):
                    candidates = sorted(candidates, key=self._site_rank)[:1]

                sites.extend(candidates)

        sites = self._deduplicate(sites)

        if not self.validation.get("retain_failed_structures", True):
            sites = [
                atoms for atoms in sites
                if atoms.info["metadata"]["geometry"]["placement_success"]
            ]

        sites.sort(key=lambda atoms: (
            tuple(atoms.info["metadata"]["al_indices"]),
            atoms.info["metadata"]["source_ring_size"],
            atoms.info["metadata"]["geometry"]["placement_score"]
        ))

        for i, atoms in enumerate(sites):
            site_id = f"{framework_code}_{metal}_M2_{i:04d}"
            atoms.info["metadata"]["site_id"] = site_id
            atoms.info["site_id"] = site_id

        return sites

    def generate(self) -> List[Atoms]:
        if not self.species["enabled"]:
            return []

        all_sites = []

        for framework_code in self.config["frameworks"]["codes"]:
            self.load_framework(framework_code)

            pairs = self.candidate_pairs()

            for metal in self.species["metals"]:
                print(f"Generating {framework_code} / {metal}²⁺")

                sites = self._generate_one(framework_code, metal, pairs)
                all_sites.extend(sites)

                print(f"  {len(sites)} structures")

        return all_sites
