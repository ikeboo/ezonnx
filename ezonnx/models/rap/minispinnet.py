from __future__ import annotations

from typing import List, Optional

import numpy as np
import onnxruntime as ort


class MiniSpinNet:
    """ONNX wrapper for MiniSpinNet local descriptor extraction.

    MiniSpinNet computes one local descriptor for each query point in
    ``keypoints`` by looking at its neighborhood in ``points``.
    """

    def __init__(
        self,
        onnx_path: str,
        patch_sample: int = 512,
        sess_options: Optional[ort.SessionOptions] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self._patch_sample = patch_sample
        self._sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._sess.get_inputs()}
        self._accepts_des_r = "des_r" in self._input_names

        output_names = [out.name for out in self._sess.get_outputs()]
        if not output_names:
            raise ValueError("MiniSpinNet ONNX model has no outputs.")
        self._output_name = "features" if "features" in output_names else output_names[0]

    def __call__(
        self,
        points: np.ndarray,
        keypoints: np.ndarray,
        des_r: Optional[float] = None,
    ) -> np.ndarray:
        """Run MiniSpinNet inference and return per-keypoint descriptors.

        Args:
            points: Reference/support point cloud with shape ``(N, 3)``.
                This is the point set MiniSpinNet uses to build local
                neighborhoods around each query point. In ``RAP``, this is
                the voxel-downsampled point cloud for one part.
            keypoints: Query points with shape ``(K, 3)``.
                MiniSpinNet computes one descriptor per point in this array.
                In ``RAP``, these are the FPS-sampled points used as the
                registration keypoints. They are expected to be in the same
                coordinate system as ``points`` and are typically a subset of
                or sampled from the same cloud.
            des_r: Optional descriptor radius passed to the ONNX model when
                the model accepts it.

        Returns:
            np.ndarray: Descriptor matrix with shape ``(K, C)``, where each
            row corresponds to one input keypoint.
        """
        ort_inputs = self._preprocess(points, keypoints, des_r)
        outputs = self._sess.run([self._output_name], ort_inputs)
        return self._postprocess(outputs)

    def _preprocess(
        self,
        points: np.ndarray,
        keypoints: np.ndarray,
        des_r: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Prepare ONNX inputs for MiniSpinNet.

        ``points`` is the support cloud and ``keypoints`` is the set of query
        points to be described. If ``points`` has fewer than ``patch_sample``
        points, it is tiled before being passed to ONNX.
        """
        pts_input = np.asarray(points, dtype=np.float32)
        kpts_input = np.asarray(keypoints, dtype=np.float32)

        if pts_input.ndim != 2 or pts_input.shape[1] != 3:
            raise ValueError("MiniSpinNet points must have shape (N, 3).")
        if kpts_input.ndim != 2 or kpts_input.shape[1] != 3:
            raise ValueError("MiniSpinNet keypoints must have shape (K, 3).")
        if len(pts_input) == 0:
            raise ValueError("MiniSpinNet requires at least one reference point.")

        if len(pts_input) < self._patch_sample:
            repeat = int(np.ceil(self._patch_sample / len(pts_input)))
            pts_input = np.tile(pts_input, (repeat, 1))[:self._patch_sample]

        ort_inputs: dict[str, np.ndarray] = {
            "pts": pts_input[np.newaxis],
            "kpts": kpts_input[np.newaxis],
        }
        if self._accepts_des_r and des_r is not None:
            ort_inputs["des_r"] = np.array([des_r], dtype=np.float32)
        return ort_inputs

    def _postprocess(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Convert MiniSpinNet outputs to a `(K, C)` feature matrix."""
        if not outputs:
            raise ValueError("MiniSpinNet inference returned no outputs.")

        features = np.asarray(outputs[0])
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        return features.astype(np.float32, copy=False)
