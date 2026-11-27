import numpy as np
import matplotlib.pyplot as plt
import nglview as nv
from ase.geometry import find_mic
from ase.atoms import Atoms
from zse.rings import get_rings
import contextlib
import io
from math import ceil
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



def show_structure(atoms, repeat=(1, 1, 1), highlight_indices=None, labels=None):
    viewer = nv.show_ase(atoms.repeat(repeat))
    viewer.clear_representations()

    viewer.add_ball_and_stick(aspectRatio=1.5)

    if highlight_indices is not None:
        indices = [int(i) for i in highlight_indices]
        n = len(atoms)
        selected = [i + k*n for k in range(int(np.prod(repeat))) for i in indices]
        selection = "@" + ",".join(map(str, selected))

        viewer.add_ball_and_stick(selection=selection, color="blue", aspectRatio=1.5)

        if labels is not None:
            if len(labels) != len(indices):
                raise ValueError("labels must have the same length as highlight_indices")

            for i, label in zip(indices, labels):
                atom_selection = "@" + ",".join(
                    str(i + k*n) for k in range(int(np.prod(repeat)))
                )
                viewer.add_label(
                    selection=atom_selection,
                    labelType="text",
                    labelText=str(label),
                    color="black",
                    zOffset=1.5
                )

    viewer.center()
    return viewer


def pair_center(atoms, i, j):
    vector = atoms.get_distance(i, j, mic=True, vector=True)
    return atoms.positions[i] + 0.5 * vector

def unwrap_ring_positions(atoms, ring_indices, view_centre=None):
    """
    Return continuous Cartesian coordinates for a ring defined by ordered atom indices.
    ring_indices should be in cyclic order around the ring.
    """
    ring_indices = [int(i) for i in ring_indices]
    pos = atoms.positions

    coords = [pos[ring_indices[0]].copy()]

    for prev, curr in zip(ring_indices[:-1], ring_indices[1:]):
        vec = atoms.get_distance(prev, curr, mic=True, vector=True)
        coords.append(coords[-1] + vec)

    coords = np.asarray(coords)

    # If a view centre is supplied, shift the whole ring so it sits near that centre
    if view_centre is not None:
        ring_centre = coords.mean(axis=0)
        shift_vec = view_centre - ring_centre
        shift_mic, _ = find_mic(shift_vec, atoms.cell, atoms.pbc)
        coords = coords + shift_mic

    return coords


def get_zse_rings(atoms, t_indices, sizes=(4, 5, 6, 7, 8)):
    rings = {}
    n = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())

    for t in sorted(set(map(int, t_indices))):
        with contextlib.redirect_stdout(io.StringIO()):
            ring_sizes, ring_paths, _, repeated = get_rings(
                atoms, index=t, max_ring=max(sizes), validation="goetzke"
            )

        repeated_symbols = np.array(repeated.get_chemical_symbols())

        for size, path in zip(ring_sizes, ring_paths):
            size = int(size)
            if size not in sizes:
                continue

            path = np.asarray(path, dtype=int)
            ids = path % n
            ring_symbols = repeated_symbols[path]

            if len(path) != 2 * size:
                continue

            if not np.array_equal(ring_symbols, symbols[ids]):
                continue

            is_t = np.isin(ring_symbols, ["Si", "Al"])
            if is_t.sum() != size or np.any(is_t == np.roll(is_t, 1)):
                continue

            coords = [repeated.positions[path[0]].copy()]

            for a, b in zip(path[:-1], path[1:]):
                vector, distance = find_mic(
                    repeated.positions[b] - repeated.positions[a],
                    atoms.cell, atoms.pbc
                )
                if distance > 2.2:
                    break
                coords.append(coords[-1] + vector)

            if len(coords) != len(path):
                continue

            coords = np.asarray(coords)

            closing, distance = find_mic(
                repeated.positions[path[0]] - repeated.positions[path[-1]],
                atoms.cell, atoms.pbc
            )
            if distance > 2.2 or np.linalg.norm(coords[-1] + closing - coords[0]) > 0.3:
                continue

            t_ids = ids[is_t].tolist()
            t_coords = coords[is_t]

            edges = np.linalg.norm(np.roll(t_coords, -1, axis=0) - t_coords, axis=1)
            if edges.max() > 4.0:
                continue

            key = (size, tuple(sorted(t_ids)))
            if key not in rings:
                rings[key] = {
                    "size": size,
                    "t_indices": t_ids,
                    "coords": t_coords
                }

    return list(rings.values())


def plot_structure_grid(structures, centers=None, center_indices=None,
                        highlight_indices=None, titles=None, radius=6.0,
                        ncols=3, figsize=None, elev=25, azim=45,
                        show_labels=False, repeat=(3, 3, 3),
                        cation_indices=None, coord_cutoff=2.7,
                        ring_overlays=False, ring_sizes=(4, 5, 6, 7, 8),
                        selected_rings_only=False,
                        framework_context=False, context_radius=12.0,):

    n = len(structures)
    nrows = ceil(n / ncols)
    figsize = figsize or (4.5 * ncols, 4.2 * nrows)
    fig = plt.figure(figsize=figsize)

    colors = {
        "Si": "#D4B04C", "Al": "#2276C7", "O": "#D95743",
        "Cu": "#B87333", "Ca": "#559B83",
        "Zn": "#7B8794", "Mn": "#9567A5", "H": "#E5E5E5"
    }
    sizes = {"O": 25, "H": 12, "Si": 48, "Al": 48}

    ring_colors = {
        4: "#22A06B",
        5: "#3E8EDE",
        6: "#E5A12B",
        7: "#A56CC1",
        8: "#D95F5F"
    }

    zse_rings = []
    if ring_overlays:
        base = structures[0].copy()
        nonframework = [a.index for a in base if a.symbol not in ("Si", "Al", "O")]
        if nonframework:
            del base[nonframework]

        for atom in base:
            if atom.symbol == "Al":
                atom.symbol = "Si"

        if framework_context:
            t_indices = [a.index for a in base if a.symbol == "Si"]
        elif highlight_indices is not None:
            t_indices = sorted({
                int(i)
                for k, atoms in enumerate(structures)
                for i in highlight_indices[k]
                if atoms[int(i)].symbol in ("Si", "Al")
            })
        else:
            t_indices = [a.index for a in base if a.symbol == "Si"]

        zse_rings = get_zse_rings(base, t_indices, sizes=ring_sizes)

        counts = {size: sum(r["size"] == size for r in zse_rings) for size in ring_sizes}
        print("ZSE identified rings:", counts)

    for k, atoms in enumerate(structures):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")

        if centers is not None:
            centre = np.asarray(centers[k], dtype=float)
        elif center_indices is not None:
            centre = atoms.positions[int(center_indices[k])]
        else:
            centre = atoms.positions.mean(axis=0)

        cell = np.asarray(atoms.cell)
        translations = [
            a*cell[0] + b*cell[1] + c*cell[2]
            for a in range(-(repeat[0]//2), repeat[0]//2 + 1)
            for b in range(-(repeat[1]//2), repeat[1]//2 + 1)
            for c in range(-(repeat[2]//2), repeat[2]//2 + 1)
        ]

        xyz, symbols, original_indices = [], [], []
        atom_symbols = np.asarray(atoms.get_chemical_symbols())

        for translation in translations:
            shifted = atoms.positions + translation - centre
            inside = np.linalg.norm(shifted, axis=1) <= radius

            xyz.extend(shifted[inside])
            symbols.extend(atom_symbols[inside])
            original_indices.extend(np.where(inside)[0])

        xyz = np.asarray(xyz).reshape(-1, 3)
        symbols = np.asarray(symbols)
        original_indices = np.asarray(original_indices, dtype=int)

        if len(xyz) == 0:
            continue

        selected = (
            set(map(int, highlight_indices[k]))
            if highlight_indices is not None else set()
        )

        if ring_overlays:
            visible_rings = []

            for ring in zse_rings:
                n_selected = len(set(ring["t_indices"]) & selected)

                # Rings containing both Al are the primary local features
                local_ring = n_selected == len(selected) and len(selected) > 0

                if not framework_context and selected_rings_only and not local_ring:
                    continue

                coords = ring["coords"].copy()
                ring_centre = coords.mean(axis=0)
                delta = centre - ring_centre
                mic_vector, _ = find_mic(delta, atoms.cell, atoms.pbc)

                coords += delta - mic_vector
                coords -= centre

                distance = np.linalg.norm(coords.mean(axis=0))
                limit = context_radius if framework_context else radius

                if distance > limit:
                    continue

                visible_rings.append((local_ring, distance, ring["size"], coords))

            visible_rings.sort(key=lambda x: (not x[0], x[1]))

            if not framework_context:
                visible_rings = visible_rings[:20]

            for local_ring, distance, size, coords in visible_rings:
                color = ring_colors.get(size, "#999999")

                # Background framework: outline only
                if framework_context and not local_ring:
                    alpha = 0.0
                    line_alpha = 0.22
                    linewidth = 0.8
                else:
                    alpha = 0.22
                    line_alpha = 0.85
                    linewidth = 1.5

                if alpha > 0:
                    rc = coords.mean(axis=0)
                    triangles = [
                        [rc, coords[i], coords[(i + 1) % len(coords)]]
                        for i in range(len(coords))
                    ]

                    ax.add_collection3d(Poly3DCollection(
                        triangles,
                        facecolors=color,
                        edgecolors="none",
                        alpha=alpha,
                        zsort="average"
                    ))

                closed = np.vstack([coords, coords[0]])
                ax.plot(
                    closed[:, 0], closed[:, 1], closed[:, 2],
                    color=color,
                    linewidth=linewidth,
                    alpha=line_alpha
                )

        t_indices = np.where(np.isin(symbols, ["Si", "Al"]))[0]
        o_indices = np.where(symbols == "O")[0]

        for i in t_indices:
            for j in o_indices:
                if np.linalg.norm(xyz[i] - xyz[j]) < 2.1:
                    ax.plot(
                        *zip(xyz[i], xyz[j]),
                        color="#89939D",
                        lw=1.1,
                        alpha=0.6
                    )

        if cation_indices is not None:
            metal_index = int(cation_indices[k])
            metal_images = np.where(original_indices == metal_index)[0]

            if len(metal_images):
                m = metal_images[np.argmin(np.linalg.norm(xyz[metal_images], axis=1))]
                metal_pos = xyz[m]
                nearest_o = {}

                for oi in o_indices:
                    original = int(original_indices[oi])
                    distance = np.linalg.norm(xyz[oi] - metal_pos)

                    if distance < coord_cutoff:
                        if original not in nearest_o or distance < nearest_o[original][0]:
                            nearest_o[original] = (distance, oi)

                for distance, oi in nearest_o.values():
                    ax.plot(
                        [metal_pos[0], xyz[oi, 0]],
                        [metal_pos[1], xyz[oi, 1]],
                        [metal_pos[2], xyz[oi, 2]],
                        linestyle="--",
                        color="#B87333",
                        linewidth=1.8,
                        alpha=0.9
                    )

        for pos, symbol, original in zip(xyz, symbols, original_indices):
            color = colors.get(symbol, "#888888")

            ax.scatter(
                *pos,
                color=color,
                s=sizes.get(symbol, 65),
                edgecolors="white",
                linewidths=0.3,
                depthshade=True
            )

            if show_labels and original in selected:
                ax.text(*(pos + 0.2), str(original), fontsize=8)

        view_radius = context_radius if framework_context else radius

        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
        ax.set_zlim(-view_radius, view_radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        if titles is not None:
            ax.set_title(titles[k], fontsize=11, pad=0)

    if ring_overlays:
        ring_handles = [
            Patch(facecolor=ring_colors[size], edgecolor=ring_colors[size], alpha=0.22,
                  label=f"{size} MR")
            for size in ring_sizes if size in ring_colors
        ]
        coord_handle = Line2D([0], [0], color="#B87333", linestyle="--",
                              linewidth=1.8, label="Metal–O coordination")
        fig.legend(
            handles=ring_handles + [coord_handle],
            loc="upper center",
            ncol=min(len(ring_handles) + 1, 6),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995)
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    return fig


def plot_generated_sites(sites: List[Atoms], **kwargs):
    metal_indices = [len(atoms) - 1 for atoms in sites]
    highlight_indices = [
        atoms.info["metadata"]["al_indices"]
        for atoms in sites
    ]
    titles = [
        f"{atoms.info['metadata']['framework']} | "
        f"{atoms.info['metadata']['cation']} | "
        f"Al {atoms.info['metadata']['al_indices']}"
        for atoms in sites
    ]

    return plot_structure_grid(
        sites,
        center_indices=metal_indices,
        cation_indices=metal_indices,
        highlight_indices=highlight_indices,
        titles=titles,
        **kwargs
    )