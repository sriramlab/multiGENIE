import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multi_gxe as mg  # noqa: E402


@pytest.fixture(scope="session")
def design():
    """A 13-environment design: one triple, one pair, nine singletons."""
    corr, blocks = mg.simulate.make_corr(seed=1)
    return corr, blocks


@pytest.fixture(scope="session")
def export(design):
    corr, _ = design
    h2 = mg.simulate.sparse_truth(corr.shape[0], tau=0.06, causal=[0, 5])
    return mg.simulate.simulate_export(corr, h2, se=0.003, seed=7)


@pytest.fixture(scope="session")
def export_json(tmp_path_factory, export):
    """The same export written to disk in the published JSON schema."""
    path = tmp_path_factory.mktemp("export") / "toy.normal_eq.json"
    payload = dict(
        schema_version=1, model=export.model, gen_nbin=1, n_env=export.n_env,
        env_names=export.env_names,
        full=dict(x_l=export.full.x_l.tolist(), y_r=export.full.y_r.tolist()),
        jackknife=[dict(x_l=j.x_l.tolist(), y_r=j.y_r.tolist())
                   for j in export.jackknife])
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture(scope="session")
def legacy():
    """The frozen pre-4.0 modules, imported the way their own scripts import them."""
    sys.path.insert(0, str(ROOT / "legacy"))
    import cluster_select
    import select_env_subset
    return select_env_subset, cluster_select
