import numpy as np
import importlib.util
from pathlib import Path


def test_training_configs_import_without_omnigibson():
    import b1k.training.config  # noqa: F401
    import serf_b1k.training.config  # noqa: F401


def test_b1k_policy_imports_without_omnigibson():
    from b1k.policies.b1k_policy import extract_state_from_proprio

    proprio = np.arange(256, dtype=np.float32)
    state = extract_state_from_proprio(proprio)

    expected = np.concatenate(
        [
            proprio[253:256],
            proprio[236:240],
            proprio[158:165],
            np.array([2.0 * (proprio[193:195].sum() / 0.1) - 1.0], dtype=np.float32),
            proprio[197:204],
            np.array([2.0 * (proprio[232:234].sum() / 0.1) - 1.0], dtype=np.float32),
        ]
    )

    np.testing.assert_allclose(state, expected)


def test_train_b1k_entrypoint_makes_omnigibson_source_importable():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "train_b1k.py"
    spec = importlib.util.spec_from_file_location("train_b1k_entrypoint", script_path)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    omnigibson_spec = importlib.util.find_spec("omnigibson")
    assert omnigibson_spec is not None
    assert str(repo_root / "BEHAVIOR-1K" / "OmniGibson") in (omnigibson_spec.origin or "")
