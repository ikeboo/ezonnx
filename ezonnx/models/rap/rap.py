#!/usr/bin/env python3
"""
ONNX-based inference for RAP (Rectified Point-Flow) point cloud registration.

Usage (CLI):
    # Register two PLY files (first = anchor/reference)
    python infer_onnx.py samples/1Cloud.ply samples/2Cloud.ply --output output/

    # Register all PLY files in a directory
    python infer_onnx.py samples/ --output output/

Dependencies: numpy, onnxruntime, scipy, open3d  (no torch/transformers)

RAP の処理概要:
    RAP は、複数の点群のうち先頭の点群をアンカー
    (参照座標系) とみなし、残りの点群をその座標系へ登録する
    ONNX 推論クラスです。処理は大きく 4 段階です。

    1. 各点群を読み込み、Torch版デモに合わせて適応的に
       ボクセルダウンサンプリング・外れ値除去・FPS を行います。
       その後、内部アンカー(最大パート)基準で正規化しつつ、
       最終出力は先頭点群を基準座標系として返します。
    2. 必要に応じて SpinNet ONNX で局所特徴を抽出します。
       SpinNet が使えない場合はゼロ特徴でフォールバックします。
    3. RAP の Flow ONNX モデルに対して Euler 積分で反復推論し、
       正規化空間で各点が登録後にあるべき位置を推定します。
    4. ダウンサンプル点の推定結果から Procrustes で剛体変換を
       求め、その変換を元のフル解像度点群へ適用し、最後に元の
       座標系へ戻して登録済み点群を出力します。
"""

from __future__ import annotations

import os
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from ...core.downloader import get_weights
from ...data_classes.registered_point_cloud import RegisteredPointCloud

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

HF_REPO_ID = "bukuroo/RAP-ONNX"
HF_FLOW_FILENAME = "rap_model_12.onnx"
HF_SPINNET_FILENAME = "mini_spinnet.onnx"


# =============================================================================
# Utility helpers
# =============================================================================


@njit(cache=True)
def _fps_numba_core(pts: np.ndarray, k: int, first_idx: int) -> np.ndarray:
    n_points = pts.shape[0]
    selected = np.empty(k, dtype=np.int64)
    selected[0] = first_idx
    dists = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        dists[i] = np.inf

    for out_idx in range(1, k):
        last_idx = selected[out_idx - 1]
        last_x = pts[last_idx, 0]
        last_y = pts[last_idx, 1]
        last_z = pts[last_idx, 2]
        for i in range(n_points):
            dx = pts[i, 0] - last_x
            dy = pts[i, 1] - last_y
            dz = pts[i, 2] - last_z
            dist = dx * dx + dy * dy + dz * dz
            if dist < dists[i]:
                dists[i] = dist

        best_idx = 0
        best_dist = dists[0]
        for i in range(1, n_points):
            if dists[i] > best_dist:
                best_dist = dists[i]
                best_idx = i
        selected[out_idx] = best_idx

    return selected


@njit(cache=True)
def _voxel_representatives_numba(
    points: np.ndarray,
    voxel_coords_sorted: np.ndarray,
    order: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, int]:
    n_points = order.shape[0]
    out = np.empty(n_points, dtype=np.int64)
    out_count = 0
    start = 0

    while start < n_points:
        vx = voxel_coords_sorted[start, 0]
        vy = voxel_coords_sorted[start, 1]
        vz = voxel_coords_sorted[start, 2]
        end = start + 1
        while end < n_points:
            if voxel_coords_sorted[end, 0] != vx:
                break
            if voxel_coords_sorted[end, 1] != vy:
                break
            if voxel_coords_sorted[end, 2] != vz:
                break
            end += 1

        cx = (vx + 0.5) * voxel_size
        cy = (vy + 0.5) * voxel_size
        cz = (vz + 0.5) * voxel_size

        best_idx = order[start]
        dx = points[best_idx, 0] - cx
        dy = points[best_idx, 1] - cy
        dz = points[best_idx, 2] - cz
        best_dist = dx * dx + dy * dy + dz * dz

        for pos in range(start + 1, end):
            idx = order[pos]
            dx = points[idx, 0] - cx
            dy = points[idx, 1] - cy
            dz = points[idx, 2] - cz
            dist = dx * dx + dy * dy + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        out[out_count] = best_idx
        out_count += 1
        start = end

    return out, out_count

def _fps(pts: np.ndarray, K: int, first_idx: Optional[int] = None) -> np.ndarray:
    """Farthest-Point Sampling.  Returns *K* indices into *pts* (N, 3)."""
    N = pts.shape[0]
    if N <= K:
        return np.arange(N)
    start_idx = 0 if first_idx is None else int(first_idx)
    if NUMBA_AVAILABLE:
        pts_contig = np.ascontiguousarray(pts.astype(np.float32, copy=False))
        return _fps_numba_core(pts_contig, K, start_idx)

    selected = np.empty(K, dtype=np.int64)
    selected[0] = start_idx
    dists = np.full(N, np.inf, dtype=np.float64)
    for i in range(1, K):
        d = np.sum((pts - pts[selected[i - 1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return selected


def _solve_procrustes(
    src: np.ndarray, tgt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm: find (R, t) that minimises ||R @ src.T + t - tgt.T||.

    Args:
        src: (N, 3)
        tgt: (N, 3)

    Returns:
        R: (3, 3) rotation matrix
        t: (3,)  translation vector
    """
    src_mean = src.mean(0)
    tgt_mean = tgt.mean(0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    H = src_c.T @ tgt_c                             # (3, 3)
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_mat @ U.T                        # (3, 3)
    t = tgt_mean - src_mean @ R.T                    # (3,)
    return R, t


def _calculate_voxel_coverage(points: np.ndarray, voxel_size: float) -> int:
    """Count occupied voxels for a point cloud."""
    if len(points) == 0:
        return 0
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    return int(len(np.unique(voxel_coords, axis=0)))


# =============================================================================
# RAP class
# =============================================================================

class RAP:
    """ONNX-based point cloud registration using the RAP flow model.

    This implementation mirrors the Torch demo/test pipeline closely:
    adaptive voxel preprocessing, voxel-adaptive FPS allocation, largest-part
    anchoring, optional SpinNet local features, Euler sampling with rigidity
    forcing, and final output expressed relative to the first input cloud.
    """

    def __init__(
        self,
        flow_model_path: Optional[str] = None,
        spinnet_path: Optional[str] = None,
        num_points: int = 500,
        num_steps: int = 10,
        des_r: Optional[float] = None,
        voxel_size: Optional[float] = None,
        voxel_ratio: float = 0.05,
        min_points_per_part: int = 200,
        max_points_per_part: int = 20000,
        adaptive_parameters: bool = True,
        remove_outliers: bool = True,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.5,
        rigidity_forcing: bool = True,
        seed: int = 42,
    ):
        self._seed = seed
        self._num_points = num_points
        self._num_steps = num_steps
        self._manual_des_r = des_r
        self._manual_voxel_size = voxel_size
        self._voxel_ratio = voxel_ratio
        self._min_points_per_part = min_points_per_part
        self._max_points_per_part = max_points_per_part
        self._adaptive_parameters = adaptive_parameters
        self._remove_outliers = remove_outliers
        self._outlier_nb_neighbors = outlier_nb_neighbors
        self._outlier_std_ratio = outlier_std_ratio
        self._rigidity_forcing = rigidity_forcing
        self._spinnet_patch_sample = 512
        self._rng = np.random.default_rng(seed)
        self.last_registered: List[np.ndarray] = []
        self.last_transforms: List[np.ndarray] = []

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        flow_model_path = self._resolve_model_asset(
            model_path=flow_model_path,
            hf_filename=HF_FLOW_FILENAME,
            label="flow",
            local_fallbacks=["./weights/rap_flow_model.onnx", "./weights/rap_model_12.onnx"],
            legacy_hf_aliases=["rap_flow_model.onnx"],
        )
        self._flow_sess = ort.InferenceSession(
            flow_model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        self._spinnet_sess: Optional[ort.InferenceSession] = None
        self._spinnet_input_names: set[str] = set()
        self._spinnet_accepts_des_r = False
        try:
            spinnet_path = self._resolve_model_asset(
                model_path=spinnet_path,
                hf_filename=HF_SPINNET_FILENAME,
                label="SpinNet",
                local_fallbacks=["./weights/mini_spinnet.onnx"],
            )
        except FileNotFoundError:
            spinnet_path = None

        if spinnet_path is not None:
            try:
                self._spinnet_sess = ort.InferenceSession(
                    spinnet_path,
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._spinnet_input_names = {
                    inp.name for inp in self._spinnet_sess.get_inputs()
                }
                self._spinnet_accepts_des_r = "des_r" in self._spinnet_input_names
            except Exception as exc:
                warnings.warn(
                    f"SpinNet ONNX could not be loaded ({exc}); "
                    "falling back to zero features."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        inputs: Sequence[Union[str, np.ndarray]],
    ) -> RegisteredPointCloud:
        """Register a list of point clouds.

        Args:
            inputs: Point cloud paths or arrays. The first cloud is the reference frame.

        Returns:
            RegisteredPointCloud: Registered point clouds and per-cloud 4x4 transforms.
        """
        point_clouds, _input_paths = self._load_inputs(inputs)
        if len(point_clouds) < 2:
            raise ValueError("At least two point clouds are required.")

        prep = self._preprocess(point_clouds)
        pred_norm = self._run_flow_model(prep)
        registered, transforms = self._postprocess(prep, pred_norm)
        self.last_registered = registered
        self.last_transforms = transforms
        return RegisteredPointCloud(
            original_data=[pts.copy() for pts in point_clouds],
            data=registered,
            translations=transforms,
        )

    def _resolve_model_asset(
        self,
        model_path: Optional[str],
        hf_filename: str,
        label: str,
        local_fallbacks: List[str],
        legacy_hf_aliases: Optional[List[str]] = None,
    ) -> str:
        legacy_hf_aliases = legacy_hf_aliases or []

        if model_path is not None:
            if os.path.exists(model_path):
                return model_path

            model_name = Path(model_path).name
            if model_name in {hf_filename, *legacy_hf_aliases}:
                return get_weights(HF_REPO_ID, hf_filename)

            raise FileNotFoundError(f"{label} model not found: {model_path}")

        for candidate in local_fallbacks:
            if os.path.exists(candidate):
                return candidate

        return get_weights(HF_REPO_ID, hf_filename)

    def _compute_auto_voxel_size(self, point_clouds: List[np.ndarray]) -> float:
        extents = []
        for pts in point_clouds:
            if len(pts) == 0:
                continue
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
            extents.append(bbox_max - bbox_min)
        if not extents:
            raise ValueError("No valid point clouds found for adaptive analysis.")

        bbox_dimensions = np.asarray(extents, dtype=np.float64)
        median_x = float(np.median(bbox_dimensions[:, 0]))
        median_y = float(np.median(bbox_dimensions[:, 1]))
        median_z = float(np.median(bbox_dimensions[:, 2]))
        median_size = float(np.median([median_x, median_y, median_z]))

        if median_size < 5.0:
            divide_factor = 200.0
        elif median_size < 30.0:
            divide_factor = 400.0
        elif median_size < 100.0:
            divide_factor = 600.0
        elif median_size < 250.0:
            divide_factor = 800.0
        elif median_size < 500.0:
            divide_factor = 1000.0
        else:
            divide_factor = 1200.0

        adaptive_voxel_size = median_size / divide_factor
        adaptive_voxel_size = max(0.0001, min(0.4, adaptive_voxel_size))
        return float(adaptive_voxel_size)

    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        if voxel_size <= 0.0 or len(points) == 0:
            return points.astype(np.float32, copy=True)
        voxel_coords = np.floor(points / voxel_size).astype(np.int64)
        order = np.lexsort((voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]))
        voxel_sorted = np.ascontiguousarray(voxel_coords[order])
        order = np.ascontiguousarray(order.astype(np.int64, copy=False))
        points_contig = np.ascontiguousarray(points.astype(np.float32, copy=False))

        if NUMBA_AVAILABLE:
            idx_all, idx_count = _voxel_representatives_numba(
                points_contig,
                voxel_sorted,
                order,
                float(voxel_size),
            )
            downsampled_indices = idx_all[:idx_count]
        else:
            downsampled_indices_list: List[int] = []
            start = 0
            while start < len(order):
                end = start + 1
                while end < len(order) and np.array_equal(voxel_sorted[end], voxel_sorted[start]):
                    end += 1
                voxel_center = (voxel_sorted[start].astype(np.float32) + 0.5) * voxel_size
                group_indices = order[start:end]
                distances = np.linalg.norm(points_contig[group_indices] - voxel_center, axis=1)
                downsampled_indices_list.append(int(group_indices[int(np.argmin(distances))]))
                start = end
            downsampled_indices = np.asarray(downsampled_indices_list, dtype=np.int64)

        return points[downsampled_indices].astype(np.float32, copy=True)

    def _batched_fps_indices(
        self,
        parts: List[np.ndarray],
        ks: List[int],
    ) -> List[np.ndarray]:
        indices_out: List[np.ndarray] = []
        for part, k in zip(parts, ks):
            if len(part) == 0 or k <= 0:
                indices_out.append(np.zeros(0, dtype=np.int64))
                continue
            first_idx = int(self._rng.integers(len(part))) if len(part) > 1 else 0
            indices_out.append(_fps(part, k, first_idx=first_idx))
        return indices_out

    def _remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        if (not self._remove_outliers or len(points) == 0 or
                len(points) < self._outlier_nb_neighbors):
            return points.astype(np.float32, copy=True)
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("open3d is required for outlier removal.") from exc

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        try:
            _filtered, inliers = pcd.remove_statistical_outlier(
                nb_neighbors=self._outlier_nb_neighbors,
                std_ratio=self._outlier_std_ratio,
            )
        except Exception:
            return points.astype(np.float32, copy=True)

        if len(inliers) == 0:
            return points.astype(np.float32, copy=True)
        return points[np.asarray(inliers, dtype=np.int64)].astype(np.float32, copy=True)

    def _compute_effective_preprocess_params(
        self,
        point_clouds: List[np.ndarray],
    ) -> Dict[str, Any]:
        if self._adaptive_parameters:
            voxel_size = self._compute_auto_voxel_size(point_clouds)
        else:
            voxel_size = self._manual_voxel_size if self._manual_voxel_size is not None else 0.25

        if self._manual_voxel_size is not None:
            voxel_size = self._manual_voxel_size

        allocated_voxel_size = 4.0 * voxel_size

        if self._manual_des_r is not None:
            des_r = self._manual_des_r
        elif self._adaptive_parameters:
            des_r = 20.0 * voxel_size
        else:
            des_r = 5.0

        voxel_ratio = self._voxel_ratio
        if self._adaptive_parameters:
            voxel_coverages = [
                _calculate_voxel_coverage(points, allocated_voxel_size)
                for points in point_clouds
                if len(points) > 0
            ]
            if not voxel_coverages:
                raise ValueError("No valid voxel coverages calculated for adaptive voxel_ratio.")
            median_voxel_coverage = float(np.median(voxel_coverages))
            current_median_point_count = median_voxel_coverage * voxel_ratio
            if current_median_point_count > self._max_points_per_part:
                voxel_ratio = self._max_points_per_part / median_voxel_coverage
                current_median_point_count = median_voxel_coverage * voxel_ratio
            if current_median_point_count < 500.0:
                voxel_ratio = 500.0 / median_voxel_coverage

        return {
            "voxel_size": float(voxel_size),
            "allocated_voxel_size": float(allocated_voxel_size),
            "des_r": float(des_r),
            "voxel_ratio": float(voxel_ratio),
        }

    def _preprocess(self, point_clouds: List[np.ndarray]) -> dict:
        params = self._compute_effective_preprocess_params(point_clouds)
        voxel_size = params["voxel_size"]
        allocated_voxel_size = params["allocated_voxel_size"]
        des_r = params["des_r"]
        voxel_ratio = params["voxel_ratio"]

        voxel_parts: List[np.ndarray] = []
        fps_source_parts: List[np.ndarray] = []
        target_per_part: List[int] = []

        for idx, pts in enumerate(point_clouds):
            voxel_pts = self._voxel_downsample(pts, voxel_size)
            voxel_parts.append(voxel_pts)

            fps_pts = self._remove_statistical_outliers(voxel_pts)
            pre_fps_cap = 20 * self._max_points_per_part
            if len(fps_pts) > pre_fps_cap:
                sub_idx = self._rng.choice(len(fps_pts), pre_fps_cap, replace=False)
                fps_pts = fps_pts[sub_idx]
            fps_source_parts.append(fps_pts.astype(np.float32, copy=True))

            if self._adaptive_parameters:
                target_points = int(_calculate_voxel_coverage(fps_pts, allocated_voxel_size) * voxel_ratio)
                target_points = max(self._min_points_per_part, target_points)
                target_points = min(target_points, len(fps_pts), self._max_points_per_part)
            else:
                target_points = min(self._num_points, len(fps_pts))

            if target_points <= 0:
                raise ValueError(f"Cloud {idx} produced zero sampled points after preprocessing.")

            target_per_part.append(int(target_points))

        fps_indices = self._batched_fps_indices(fps_source_parts, target_per_part)
        sampled_parts: List[np.ndarray] = []
        for idx, (fps_pts, fps_idx) in enumerate(zip(fps_source_parts, fps_indices)):
            sampled = fps_pts[fps_idx].astype(np.float32, copy=True)
            sampled_parts.append(sampled)

        counts = [len(part) for part in sampled_parts]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        n_parts = len(sampled_parts)
        primary_idx = int(np.argmax(counts))
        primary_trans = sampled_parts[primary_idx].mean(axis=0, dtype=np.float64)

        shifted = [
            sampled.astype(np.float64) - primary_trans
            for sampled in sampled_parts
        ]
        scale = float(np.max(np.abs(shifted[primary_idx])) * 1.5)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Invalid normalization scale computed from input point clouds.")

        scaled = [part / scale for part in shifted]
        all_scaled = np.concatenate(scaled, axis=0)
        gt_trans = all_scaled.mean(axis=0)
        pts_gt_parts = [part - gt_trans for part in scaled]

        cond_parts: List[np.ndarray] = []
        part_trans: List[np.ndarray] = []
        for i, pts_gt_part in enumerate(pts_gt_parts):
            trans_i = pts_gt_part.mean(axis=0)
            part_trans.append(trans_i.astype(np.float64))
            if i == primary_idx:
                cond_parts.append((pts_gt_part + gt_trans).astype(np.float32))
            else:
                cond_parts.append((pts_gt_part - trans_i).astype(np.float32))

        cond = np.concatenate(cond_parts, axis=0).astype(np.float32)
        segment_ids = np.empty(offsets[-1], dtype=np.int64)
        anchor_indices = np.zeros(offsets[-1], dtype=bool)
        for i in range(n_parts):
            segment_ids[offsets[i]:offsets[i + 1]] = i
        anchor_indices[offsets[primary_idx]:offsets[primary_idx + 1]] = True

        local_features = self._extract_features(
            voxel_parts=voxel_parts,
            sampled_parts=sampled_parts,
            offsets=offsets,
            des_r=des_r,
        )

        return {
            "cond": cond,
            "local_features": local_features,
            "scale": np.array([scale], dtype=np.float32),
            "anchor_indices": anchor_indices,
            "segment_ids": segment_ids,
            "counts": counts,
            "offsets": offsets,
            "primary_idx": primary_idx,
            "primary_trans": primary_trans.astype(np.float64),
            "scale_val": scale,
            "gt_trans": gt_trans.astype(np.float64),
            "part_trans": part_trans,
            "parts_orig": [pts.astype(np.float32, copy=True) for pts in point_clouds],
            "sampled_parts": sampled_parts,
            "target_per_part": target_per_part,
            "effective_params": params,
        }

    def _extract_features(
        self,
        voxel_parts: List[np.ndarray],
        sampled_parts: List[np.ndarray],
        offsets: np.ndarray,
        des_r: float,
    ) -> np.ndarray:
        """Extract per-point SpinNet descriptors if ONNX model is available."""
        total_points = int(offsets[-1])
        feat_out = np.zeros((total_points, 32), dtype=np.float32)
        if self._spinnet_sess is None:
            return feat_out

        for i, (pts_ref, kpts) in enumerate(zip(voxel_parts, sampled_parts)):
            s, e = offsets[i], offsets[i + 1]
            if len(kpts) == 0:
                continue
            try:
                pts_input = pts_ref.astype(np.float32, copy=False)
                if len(pts_input) < self._spinnet_patch_sample:
                    repeat = int(np.ceil(self._spinnet_patch_sample / max(len(pts_input), 1)))
                    pts_input = np.tile(pts_input, (repeat, 1))[:self._spinnet_patch_sample]

                ort_inputs: Dict[str, Any] = {
                    "pts": pts_input[np.newaxis],
                    "kpts": kpts.astype(np.float32, copy=False)[np.newaxis],
                }
                if self._spinnet_accepts_des_r:
                    ort_inputs["des_r"] = np.array([des_r], dtype=np.float32)

                feats = np.asarray(self._spinnet_sess.run(["features"], ort_inputs)[0])
                if feats.ndim == 3 and feats.shape[0] == 1:
                    feats = feats[0]
                feat_out[s:e] = feats.astype(np.float32, copy=False)
            except Exception:
                pass
        return feat_out

    def _rigidify_prediction(
        self,
        prediction: np.ndarray,
        condition: np.ndarray,
        counts: List[int],
    ) -> np.ndarray:
        rigid = np.zeros_like(prediction, dtype=np.float32)
        offset = 0
        for count in counts:
            if count <= 0:
                continue
            src = condition[offset:offset + count].astype(np.float64)
            tgt = prediction[offset:offset + count].astype(np.float64)
            rot, trans = _solve_procrustes(src, tgt)
            rigid[offset:offset + count] = (src @ rot.T + trans).astype(np.float32)
            offset += count
        return rigid

    def _run_flow_model(self, prep: dict) -> np.ndarray:
        cond = prep["cond"]
        local_features = prep["local_features"]
        scale = prep["scale"]
        anchor_indices = prep["anchor_indices"]
        segment_ids = prep["segment_ids"]
        counts = prep["counts"]

        x_1 = self._rng.standard_normal(cond.shape).astype(np.float32)
        x_t = x_1.copy()
        last_x_0_hat = x_t.copy()
        dt = 1.0 / self._num_steps

        for step in range(self._num_steps):
            t = float(1.0 - step * dt)
            timestep = np.array([t], dtype=np.float32)
            v = self._flow_sess.run(
                ["velocity"],
                {
                    "x": x_t,
                    "timestep": timestep,
                    "cond_coord": cond,
                    "local_features": local_features,
                    "scale": scale,
                    "anchor_indices": anchor_indices,
                    "segment_ids": segment_ids,
                },
            )[0].astype(np.float32)

            x_0_hat = x_t - v * t
            last_x_0_hat = x_0_hat
            x_t = x_t - dt * v

            if self._rigidity_forcing:
                x_0_hat_rigid = self._rigidify_prediction(x_0_hat, cond, counts)
                x_t = x_0_hat_rigid * (1.0 - t + dt) + x_1 * (t - dt)

        return last_x_0_hat.astype(np.float32)

    def _postprocess(
        self, prep: dict, pred_norm: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        offsets = prep["offsets"]
        primary_idx = prep["primary_idx"]
        primary_trans = prep["primary_trans"]
        scale_val = prep["scale_val"]
        gt_trans = prep["gt_trans"]
        part_trans = prep["part_trans"]
        parts_orig = prep["parts_orig"]
        cond = prep["cond"]

        world_transforms: List[np.ndarray] = []
        for i, _pts_orig in enumerate(parts_orig):
            s, e = offsets[i], offsets[i + 1]
            cond_i = cond[s:e].astype(np.float64)
            pred_i = pred_norm[s:e].astype(np.float64)
            rot_i, trans_i = _solve_procrustes(cond_i, pred_i)

            if i == primary_idx:
                trans_world = (
                    (-primary_trans) @ rot_i.T
                    + scale_val * (trans_i + gt_trans)
                    + primary_trans
                )
            else:
                trans_world = (
                    (-primary_trans - scale_val * (gt_trans + part_trans[i])) @ rot_i.T
                    + scale_val * (trans_i + gt_trans)
                    + primary_trans
                )

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rot_i
            transform[:3, 3] = trans_world
            world_transforms.append(transform)

        reference_inv = np.linalg.inv(world_transforms[0])
        registered: List[np.ndarray] = [parts_orig[0].copy()]
        transforms: List[np.ndarray] = [np.eye(4, dtype=np.float64)]

        for i in range(1, len(parts_orig)):
            relative = reference_inv @ world_transforms[i]
            pts = parts_orig[i].astype(np.float64)
            reg = pts @ relative[:3, :3].T + relative[:3, 3]
            registered.append(reg.astype(np.float32))
            transforms.append(relative.astype(np.float64, copy=False))

        return registered, transforms

    def _save(
        self,
        arrays: List[np.ndarray],
        paths: List[str],
        output_dir: str = ".",
    ) -> List[str]:
        """Save registered clouds as PLY files.

        The anchor cloud (index 0) is *not* saved because it is unchanged.
        Output filenames are ``{stem}_reg.ply``.

        Args:
            arrays     : List of ``(N_i, 3)`` registered clouds.
            paths      : Original input file paths (same order as arrays).
                         Non-anchor entries must be valid paths.
            output_dir : Directory where output files are written.

        Returns:
            List of output file paths (only non-anchor clouds).
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for _save. "
                              "Install it with: pip install open3d")

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        for i, (arr, src_path) in enumerate(zip(arrays, paths)):
            if i == 0:
                continue  # skip anchor

            stem = Path(src_path).stem if src_path else f"cloud_{i}"
            out_path = os.path.join(output_dir, f"{stem}_reg.ply")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(arr.astype(np.float64))
            o3d.io.write_point_cloud(out_path, pcd)
            saved_paths.append(out_path)

        return saved_paths

    def _save_transforms(
        self,
        transforms: List[np.ndarray],
        paths: List[str],
        output_dir: str = ".",
    ) -> List[str]:
        """Save per-cloud 4x4 transforms as text files.

        Each transform maps original input coordinates into the registered
        output coordinates expressed in the first input cloud's frame.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        for i, transform in enumerate(transforms):
            src_path = paths[i] if i < len(paths) else ""
            stem = Path(src_path).stem if src_path else f"cloud_{i}"
            out_path = os.path.join(output_dir, f"{stem}_transform.txt")
            np.savetxt(
                out_path,
                np.asarray(transform, dtype=np.float64),
                fmt="%.10f",
                header=(
                    "4x4 transform from the original input cloud to the "
                    "registered output in the first input cloud frame."
                ),
            )
            saved_paths.append(out_path)

        return saved_paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_inputs(
        self, inputs: Sequence[Union[str, np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Return (list-of-arrays, list-of-paths)."""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for loading PLY files. "
                              "Install it with: pip install open3d")

        arrays: List[np.ndarray] = []
        paths: List[str] = []

        for item in inputs:
            if isinstance(item, np.ndarray):
                arrays.append(item.astype(np.float32))
                paths.append("")
            elif isinstance(item, str):
                pcd = o3d.io.read_point_cloud(item)
                pts = np.asarray(pcd.points, dtype=np.float32)
                if pts.shape[0] == 0:
                    raise ValueError(f"Empty point cloud: {item}")
                arrays.append(pts)
                paths.append(item)
            else:
                raise TypeError(f"Unsupported input type: {type(item)}")

        return arrays, paths


# =============================================================================
# CLI entry point
# =============================================================================

def _collect_ply_files(args_inputs: List[str]) -> List[str]:
    """Expand directory arguments to sorted PLY file lists."""
    from natsort import natsorted

    files: List[str] = []
    for item in args_inputs:
        p = Path(item)
        if p.is_dir():
            found = natsorted(str(f) for f in p.glob("*.ply"))
            if not found:
                raise ValueError(f"No PLY files found in directory: {item}")
            files.extend(found)
        elif p.suffix.lower() == ".ply":
            files.append(str(p))
        else:
            raise ValueError(f"Expected a PLY file or directory, got: {item}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAP ONNX inference — register PLY point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="PLY file paths or a directory.  The FIRST file is the anchor.",
    )
    parser.add_argument(
        "--output", default="output/",
        help="Directory to write registered PLY files.",
    )
    parser.add_argument(
        "--flow_model", default=None,
        help="Path to ONNX flow model. If omitted, it is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--spinnet", default=None,
        help="Path to ONNX SpinNet model. If omitted, it is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--num_points", type=int, default=500,
        help="Fallback FPS points per part when adaptive preprocessing is disabled.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="Euler integration steps.",
    )
    parser.add_argument(
        "--des_r", type=float, default=None,
        help="SpinNet descriptor radius. If omitted, adaptive mode uses 20 * voxel_size.",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=None,
        help="Voxel size for preprocessing. If omitted, adaptive mode chooses it from bbox size.",
    )
    parser.add_argument(
        "--voxel_ratio", type=float, default=0.05,
        help="Voxel-adaptive FPS ratio.",
    )
    parser.add_argument(
        "--min_points_per_part", type=int, default=200,
        help="Minimum sampled points per part in adaptive mode.",
    )
    parser.add_argument(
        "--max_points_per_part", type=int, default=20000,
        help="Maximum sampled points per part in adaptive mode.",
    )
    parser.add_argument(
        "--adaptive_parameters", action="store_true", default=True,
        help="Use Torch demo-style adaptive preprocessing parameters.",
    )
    parser.add_argument(
        "--no_adaptive_parameters", dest="adaptive_parameters", action="store_false",
        help="Disable adaptive preprocessing and use fixed num_points/voxel_size/des_r.",
    )
    parser.add_argument(
        "--remove_outliers", action="store_true", default=True,
        help="Apply statistical outlier removal before FPS.",
    )
    parser.add_argument(
        "--no_remove_outliers", dest="remove_outliers", action="store_false",
        help="Disable statistical outlier removal.",
    )
    parser.add_argument(
        "--outlier_nb_neighbors", type=int, default=20,
        help="Neighbor count for statistical outlier removal.",
    )
    parser.add_argument(
        "--outlier_std_ratio", type=float, default=2.5,
        help="Std ratio for statistical outlier removal.",
    )
    parser.add_argument(
        "--rigidity_forcing", action="store_true", default=True,
        help="Use RAP rigidity forcing during Euler sampling.",
    )
    parser.add_argument(
        "--no_rigidity_forcing", dest="rigidity_forcing", action="store_false",
        help="Disable rigidity forcing.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initial noise.",
    )
    args = parser.parse_args()

    # Collect PLY files
    ply_files = _collect_ply_files(args.inputs)
    if len(ply_files) < 2:
        parser.error("At least two PLY files are required.")

    # Build model
    model = RAP(
        flow_model_path=args.flow_model,
        spinnet_path=args.spinnet,
        num_points=args.num_points,
        num_steps=args.num_steps,
        des_r=args.des_r,
        voxel_size=args.voxel_size,
        voxel_ratio=args.voxel_ratio,
        min_points_per_part=args.min_points_per_part,
        max_points_per_part=args.max_points_per_part,
        adaptive_parameters=args.adaptive_parameters,
        remove_outliers=args.remove_outliers,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        rigidity_forcing=args.rigidity_forcing,
        seed=args.seed,
    )

    # Run registration
    result = model(ply_files)

    # Save results
    model._save(result.data, ply_files, output_dir=args.output)
    model._save_transforms(
        result.translations,
        ply_files,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
