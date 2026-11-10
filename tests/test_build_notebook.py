import importlib.util
import re
import shutil

import nbformat
import pytest

from per3clock import config

SCRIPT = config.ROOT / "scripts" / "build_notebook.py"
spec = importlib.util.spec_from_file_location("build_notebook", SCRIPT)
build_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebook)


def _stage_numbers(nb):
    nums = []
    for c in nb.cells:
        if c.cell_type == "markdown":
            m = re.match(r"^# Stage (\d)\b", c.source)
            if m:
                nums.append(int(m.group(1)))
    return nums


def test_build_produces_valid_ordered_notebook(tmp_path):
    out = tmp_path / "final.ipynb"
    assert build_notebook.main(["-o", str(out)]) == 0
    nb = nbformat.read(out, as_version=4)
    nbformat.validate(nb)
    assert _stage_numbers(nb) == list(range(1, 9))
    code = [c for c in nb.cells if c.cell_type == "code"]
    boots = [c for c in nb.cells if "bootstrap" in c.metadata.get("tags", [])]
    assert len(boots) == 1 and code[0] is boots[0]
    assert "google.colab" in boots[0].source
    assert all(not c.get("outputs") for c in code)


def test_every_stage_has_bootstrap_first_and_heading():
    for p in build_notebook.stage_paths():
        nb = nbformat.read(p, as_version=4)
        assert "bootstrap" in nb.cells[0].metadata.get("tags", []), p.name
        assert any(c.cell_type == "markdown" and c.source.startswith("# Stage") for c in nb.cells)
        assert any("Interpretation" in c.source for c in nb.cells if c.cell_type == "markdown")


def test_stage_notebooks_have_no_outputs():
    for p in build_notebook.stage_paths():
        nb = nbformat.read(p, as_version=4)
        for c in nb.cells:
            if c.cell_type == "code":
                assert not c.outputs, f"{p.name}: clear outputs before committing"


def test_missing_stage_fails(tmp_path):
    for p in build_notebook.stage_paths()[:-1]:
        shutil.copy(p, tmp_path / p.name)
    with pytest.raises(SystemExit):
        build_notebook.build(tmp_path)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", ["01_question.ipynb", "02_data_acquisition.ipynb"])
def test_stage_notebook_executes(name):
    from nbclient import NotebookClient

    path = config.ROOT / "notebooks" / "stages" / name
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
