"""Check release versions without importing MOSA or installing dependencies."""

import argparse
import ast
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check_source(root: Path, tag: str) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project["name"] != "mosalib":
        raise ValueError("Expected the PyPI project name mosalib")
    version = project["version"]
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Release versions must use X.Y.Z (stable releases only)")
    if tag and tag != f"v{version}":
        raise ValueError(f"Expected tag v{version}, got {tag}")
    tree = ast.parse((root / "src/mosa/__init__.py").read_text())
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if versions != [version]:
        raise ValueError("mosa.__version__ must match pyproject.toml")
    return version


def check_distributions(dist: Path, version: str) -> None:
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1 or len(list(dist.iterdir())) != 2:
        raise ValueError("Expected exactly one wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as archive:
        paths = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
        if len(paths) != 1:
            raise ValueError("Expected one wheel METADATA file")
        wheel_metadata = archive.read(paths[0])
    with tarfile.open(sdists[0]) as archive:
        paths = [
            m
            for m in archive.getmembers()
            if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
        ]
        if len(paths) != 1:
            raise ValueError("Expected one top-level source PKG-INFO file")
        stream = archive.extractfile(paths[0])
        if stream is None:
            raise ValueError("Source PKG-INFO must be a regular file")
        with stream:
            sdist_metadata = stream.read()
    for metadata in (wheel_metadata, sdist_metadata):
        fields = BytesParser().parsebytes(metadata)
        if fields["Name"] != "mosalib" or fields["Version"] != version:
            raise ValueError(f"Distribution metadata must identify mosalib {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    try:
        version = check_source(Path.cwd(), args.tag)
        if args.dist:
            check_distributions(args.dist, version)
    except (ValueError, KeyError) as exc:
        parser.exit(1, f"Release validation failed: {exc}\n")
    print(f"Validated mosalib {version}")
