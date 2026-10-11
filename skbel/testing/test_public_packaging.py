#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Public packaging checks: exports, wheel contents policy, README quickstart, versions."""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.rst"
CHANGELOG = ROOT / "CHANGELOG.md"
CITATION = ROOT / "CITATION.cff"


SEMVER = re.compile(r"\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.+-]+)?")


def _require(path):
    if not PYPROJECT.is_file():
        pytest.skip("source-only check: pyproject.toml is not available")
    assert path.is_file(), f"{path.name} is missing from the source checkout"
    return path.read_text(encoding="utf-8")


def _toml_section(text, header):
    match = re.search(rf"^\[{re.escape(header)}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert match, f"missing [{header}] in pyproject.toml"
    return match.group(1)


def _normalize(version):
    return version.strip().strip("\"'").removeprefix("v")


def test_exports_and_wildcard_access():
    import skbel

    for name in ("BEL", "design", "evaluation", "learning", "metrics"):
        assert name in skbel.__all__

    namespace = {}
    exec("from skbel import *", namespace)
    for name in skbel.__all__:
        assert name in namespace, name
    assert namespace["BEL"] is skbel.BEL
    assert namespace["design"] is skbel.design
    assert namespace["evaluation"] is skbel.evaluation
    assert namespace["learning"] is skbel.learning
    assert namespace["metrics"] is skbel.metrics


def test_import_does_not_load_torch():
    code = "import sys, skbel; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wheel_excludes_test_arrays_and_source_keeps_fixtures():
    text = _require(PYPROJECT)
    wheel = _toml_section(text, "tool.hatch.build.targets.wheel")
    assert "skbel/testing/*.npy" in wheel
    sdist = _toml_section(text, "tool.hatch.build.targets.sdist")
    assert '"skbel/"' in sdist
    assert "exclude" not in sdist
    assert (ROOT / "skbel" / "testing" / "ref_mean.npy").is_file()


def test_pyproject_has_no_personal_email():
    text = _require(PYPROJECT)
    assert "email" not in text


def _readme_quickstart():
    lines = _require(README).splitlines()
    heading = [
        i
        for i, line in enumerate(lines[:-1])
        if line == "Quickstart" and re.fullmatch(r"[~=-]{3,}", lines[i + 1])
    ]
    assert heading, "README.rst has no Quickstart section"

    index = heading[0] + 2
    while index < len(lines) and not lines[index].endswith("::"):
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        assert not (lines[index] and re.fullmatch(r"[~=-]{3,}", nxt)), (
            "Quickstart has no literal code block"
        )
        index += 1
    assert index < len(lines), "Quickstart has no literal code block"

    block = []
    for line in lines[index + 1 :]:
        if line.strip() and not line.startswith("    "):
            break
        block.append(line)
    code = textwrap.dedent("\n".join(block)).strip()
    assert code, "Quickstart literal code block is empty"
    return code + "\n"


def test_readme_quickstart_is_finite_and_deterministic():
    code = _readme_quickstart()
    runs = []
    for _ in range(2):
        namespace = {}
        exec(compile(code, "README.rst", "exec"), namespace)
        runs.append(namespace["samples"])
    first, second = runs
    assert first.shape == (1, 50, 6)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_pyproject_citation_and_changelog_versions_agree():
    declared = re.search(r'^version\s*=\s*"([^"]+)"', _require(PYPROJECT), re.M)
    assert declared, "pyproject.toml has no version"
    pyproject = _normalize(declared.group(1))
    assert SEMVER.fullmatch(pyproject), pyproject

    changelog = _require(CHANGELOG)
    released = re.findall(r"^##\s+\[?v?(\d+\.\d+\.\d+[^\s\]]*)\]?", changelog, re.M)
    assert released, "CHANGELOG.md has no released version heading"
    assert SEMVER.fullmatch(_normalize(released[0])), released[0]
    assert pyproject == _normalize(released[0])

    citation = re.search(r"^version:\s*(\S+)", _require(CITATION), re.M)
    assert citation, "CITATION.cff has no version"
    assert SEMVER.fullmatch(_normalize(citation.group(1))), citation.group(1)
    assert pyproject == _normalize(citation.group(1))
