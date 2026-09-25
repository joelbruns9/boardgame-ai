"""The checkpoint the scheduler sweeps measure: the run's OWN starting network.

Both sweeps load their checkpoint strictly, as a resume would, so a warm-start
source from before an encoder change fails them outright ("checkpoint migration
required"), and a source without the run's new modules (W1/W2/W5) would load as
a lighter network than the run serves -- a cheaper forward pass, and a geometry
ranked for a model nobody is running.

This writes the file the run itself starts from: the manifest's configuration,
seeded through the same `initialize_learner` path `--init-checkpoint` takes, so
the migration and module set are the run's by construction rather than
re-derived here.

    python -m games.seven_wonders_duel.sweep_checkpoint \
        --checkpoint SRC.pt --manifest run_config.json --output OUT.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

from . import phase_d as pd
from .f4_phase_d_sweep import config_from_manifest, geometry_from_checkpoint
from .training_adapter import SevenWondersDuelLifecycleAdapter


def write_sweep_checkpoint(
    checkpoint: Path, manifest: Path, output: Path, *, precision: str = "bf16"
) -> Path:
    work = output.parent / (output.stem + "_work")
    config = config_from_manifest(
        manifest,
        output=work,
        device="cpu",
        games=1,
        precision=precision,
        geometry=geometry_from_checkpoint(str(checkpoint)),
    )
    config = dataclasses.replace(config, init_checkpoint=str(checkpoint))
    loop = pd.PhaseDLoop(config)
    loop.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact = SevenWondersDuelLifecycleAdapter(loop).initialize_learner(
        seed=config.seed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact.path, output)
    shutil.rmtree(work, ignore_errors=True)
    # Proof, not hope: the sweeps load it strictly, so load it strictly here.
    loop.load_model(output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args(argv)
    path = write_sweep_checkpoint(
        args.checkpoint, args.manifest, args.output, precision=args.precision
    )
    print(f"sweep checkpoint: {path}")


if __name__ == "__main__":
    main()
