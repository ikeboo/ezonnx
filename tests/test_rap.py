from types import SimpleNamespace

import numpy as np

from ezonnx.data_classes.registered_point_cloud import RegisteredPointCloud
from ezonnx.models.rap import rap as rap_module


class _DummySession:
    def __init__(self, model_path: str, *args, **kwargs):
        self.model_path = model_path

    def get_inputs(self):
        if self.model_path.endswith("mini_spinnet.onnx"):
            return [
                SimpleNamespace(name="pts"),
                SimpleNamespace(name="kpts"),
                SimpleNamespace(name="des_r"),
            ]
        return []


def test_rap_downloads_models_from_hf_when_paths_are_not_provided(monkeypatch):
    download_calls: list[tuple[str, str]] = []

    def fake_get_weights(repo_id: str, filename: str) -> str:
        download_calls.append((repo_id, filename))
        return f"/tmp/{filename}"

    monkeypatch.setattr(rap_module, "get_weights", fake_get_weights)
    monkeypatch.setattr(rap_module.ort, "InferenceSession", _DummySession)
    monkeypatch.setattr(rap_module.os.path, "exists", lambda _path: False)

    model = rap_module.RAP()

    assert download_calls == [
        (rap_module.HF_REPO_ID, rap_module.HF_FLOW_FILENAME),
        (rap_module.HF_REPO_ID, rap_module.HF_SPINNET_FILENAME),
    ]
    assert model._flow_sess.model_path.endswith(rap_module.HF_FLOW_FILENAME)
    assert model._spinnet_sess is not None


def test_rap_call_returns_registered_point_cloud(monkeypatch):
    def fake_get_weights(repo_id: str, filename: str) -> str:
        return f"/tmp/{filename}"

    monkeypatch.setattr(rap_module, "get_weights", fake_get_weights)
    monkeypatch.setattr(rap_module.ort, "InferenceSession", _DummySession)
    monkeypatch.setattr(rap_module.os.path, "exists", lambda _path: False)

    model = rap_module.RAP()
    point_clouds = [
        np.zeros((4, 3), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
    ]
    registered = [
        point_clouds[0].copy(),
        (point_clouds[1] + 2.0).astype(np.float32),
    ]
    transforms = [
        np.eye(4, dtype=np.float64),
        np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ]

    monkeypatch.setattr(model, "_load_inputs", lambda inputs: (point_clouds, ["", ""]))
    monkeypatch.setattr(model, "_preprocess", lambda pcs: {"cond": np.zeros((8, 3), dtype=np.float32)})
    monkeypatch.setattr(model, "_run_flow_model", lambda prep: np.zeros((8, 3), dtype=np.float32))
    monkeypatch.setattr(model, "_postprocess", lambda prep, pred: (registered, transforms))

    result = model(point_clouds)

    assert isinstance(result, RegisteredPointCloud)
    assert len(result.data) == 2
    assert len(result.original_data) == 2
    assert len(result.translations) == 2
    np.testing.assert_allclose(result.original_data[0], point_clouds[0])
    np.testing.assert_allclose(result.original_data[1], point_clouds[1])
    np.testing.assert_allclose(result.data[0], registered[0])
    np.testing.assert_allclose(result.data[1], registered[1])
    np.testing.assert_allclose(result.translations[1], transforms[1])
    assert result.original_img.size == 0


def test_rap_preprocess_helpers_are_torch_free():
    source = rap_module.RAP._voxel_downsample.__code__.co_names
    fps_source = rap_module.RAP._batched_fps_indices.__code__.co_names

    assert "torch" not in source
    assert "torch" not in fps_source
