import yaml
import pandas as pd
from pathlib import Path
from copy import deepcopy
from ase.io import write

from src.structures.m2plus import M2PlusGenerator

with open("configs/structure_generation.yaml") as f:
    config = yaml.safe_load(f)

output = Path("data/generated_structures/m2plus")
output.mkdir(parents=True, exist_ok=True)
summaries_dir = Path("data/structure_summaries")
summaries_dir.mkdir(parents=True, exist_ok=True)

generator = M2PlusGenerator(config)

all_diagnostics = []
failures = []

for framework in config["frameworks"]["codes"]:
    try:
        print(f"\nGenerating {framework}")
        generator.load_framework(framework)

        # Enumerate rings and Al pairs once per framework
        pairs = generator.candidate_pairs()

        for metal in config["species"]["M2+"]["metals"]:
            print(f"  {metal}")

            try:
                structures = generator._generate_one(framework, metal, pairs)
                if not structures:
                    continue

                # Save all configurations for this framework-metal pair
                clean_structures = []

                for atoms in structures:
                    structure = atoms.copy()
                    structure.info = {"site_id": atoms.info["metadata"]["site_id"]}
                    clean_structures.append(structure)

                write(output / f"{framework}_{metal}.traj", clean_structures)

                # Save diagnostic metadata separately
                all_diagnostics.append(generator.summary(structures))

            except Exception as e:
                failures.append({
                    "framework": framework,
                    "metal": metal,
                    "error": str(e)
                })
                print(f"    Failed: {e}")

    except Exception as e:
        failures.append({
            "framework": framework,
            "metal": None,
            "error": str(e)
        })
        print(f"  Framework failed: {e}")

if all_diagnostics:
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    diagnostics.to_csv(summaries_dir / "m2plus_summary.csv", index=False)
    print(f"\nSaved {len(diagnostics)} structures")

if failures:
    pd.DataFrame(failures).to_csv(summaries_dir / "m2plus_failures.csv", index=False)

with open(output / "generation_config.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)