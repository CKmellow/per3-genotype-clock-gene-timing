"""The live-demo notebook stays valid Python and points at real subjects and genes."""
import ast
import json
import re

from per3clock import config, data

DEMO = config.ROOT / "notebooks" / "PER3_live_demo.ipynb"


def _code_cells():
    nb = json.loads(DEMO.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_demo_cells_parse():
    for src in _code_cells():
        ast.parse(src)


def test_demo_dropdowns_use_real_values():
    subjects = set(data.analysis_samples(for_phase_analysis=True)["subject"])
    field = re.compile(r'(\w+) = "([^"]+)"\s+# @param (\[.*\])')
    params = [m for src in _code_cells() for m in field.findall(src)]
    assert params, "no Colab form fields found"
    for name, default, options in params:
        opts = json.loads(options)
        assert default in opts, name
        if name == "PERSON":
            assert set(opts) <= subjects
        if name == "GENE":
            assert set(opts) <= set(config.CLOCK_GENES)
