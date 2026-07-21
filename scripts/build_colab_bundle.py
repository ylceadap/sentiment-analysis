"""Build a minimal current-worktree source bundle for GPU experiments in Colab."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

ALLOWED_ROOTS = {
    "configs",
    "src",
    "scripts",
    "tests",
    "notebooks",
    "pyproject.toml",
    "README.md",
    "Python_Engineer_Challenge_2.csv",
}


def tracked_and_untracked_files() -> list[Path]:
    """List non-ignored repository files, including current uncommitted additions."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(value.decode()) for value in result.stdout.split(b"\0") if value]


def build_bundle(destination: Path) -> list[Path]:
    """Write only source, configuration, notebook, and immutable CSV inputs to a ZIP."""
    selected = [
        path
        for path in tracked_and_untracked_files()
        if path.is_file() and path.parts[0] in ALLOWED_ROOTS and path.name != destination.name
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.as_posix())
    return selected


def main() -> None:
    """Build the upload archive and print a compact manifest summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("robbert_colab_source.zip"))
    args = parser.parse_args()
    files = build_bundle(args.output)
    print(f"Wrote {args.output} with {len(files)} files ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
