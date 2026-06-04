"""Import-hygiene regression tests for the data_structure circular cluster.

TensorLogic, Operators and Category form an import cycle
(TensorLogic -> Operators -> Category -> TensorLogic).  Each must be importable
as the FIRST module loaded (the cycle entry point).  A fresh interpreter per
case is required: once a module is in sys.modules, import order no longer
matters, so the regression only reproduces in a clean process.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", [
    "data_structure.TensorLogic",
    "data_structure.Category",
    "data_structure.Operators",
    "data_structure.TensorDSL",
])
def test_module_importable_as_cycle_entry(module):
    """Importing any cycle member first must not raise (partial-init ImportError)."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_category_reexports_tensor_logic_names():
    """The lazy re-export still exposes TensorEquation/TensorProgram via Category."""
    code = (
        "import data_structure.Category as cat; "
        "from data_structure.TensorLogic import TensorEquation, TensorProgram; "
        "assert cat.TensorEquation is TensorEquation; "
        "assert cat.TensorProgram is TensorProgram"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
