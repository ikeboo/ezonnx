from __future__ import annotations

from importlib import import_module
from typing import Any, ClassVar, List, Tuple

import cv2
import numpy as np
from pydantic import ConfigDict, Field

from .result import Result


class RegisteredPointCloudResult(Result):
	"""Data class for point cloud registration results.

	Attributes:
		data (List[Any]): Registered point clouds. The first point cloud is the fixed reference.
		translations (List[np.ndarray]): Rigid transformation matrices in shape (4, 4).
			The first transform corresponds to the fixed reference point cloud.
		image_size (Tuple[int, int]): Output image size as (width, height).
		point_size (int): Point size used for visualization.
		background_color (Tuple[int, int, int]): Background color in BGR.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	original_img: np.ndarray = Field(
		default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
	)
	data: List[Any]
	translations: List[np.ndarray]
	image_size: Tuple[int, int] = (1024, 1024)
	point_size: int = 3
	background_color: Tuple[int, int, int] = (255, 255, 255)

	_OPEN3D_MODULE: ClassVar[Any | None] = None
	_PALETTE: ClassVar[List[Tuple[int, int, int]]] = [
		(255, 99, 71),
		(60, 179, 113),
		(65, 105, 225),
		(255, 165, 0),
		(186, 85, 211),
		(64, 224, 208),
		(220, 20, 60),
		(255, 215, 0),
	]

	def _vizualize(self) -> np.ndarray:
		"""Visualize registered point clouds from a top-down viewpoint.

		Open3D is imported lazily on the first visualization call.

		Returns:
			np.ndarray: Visualized image in shape (H, W, 3). BGR
		"""
		self._validate_inputs()
		o3d = self._get_open3d()
		point_clouds = self._build_point_clouds(o3d)

		try:
			return self._render_with_open3d(point_clouds, o3d)
		except Exception:
			return self._render_top_view(point_clouds)

	@classmethod
	def _get_open3d(cls) -> Any:
		"""Import and cache the Open3D module on demand."""
		if cls._OPEN3D_MODULE is None:
			try:
				cls._OPEN3D_MODULE = import_module("open3d")
			except ImportError as exc:
				raise ImportError(
					"RegisteredPointCloudResult.visualized_img requires open3d. "
					"Please install open3d to visualize registered point clouds."
				) from exc
		return cls._OPEN3D_MODULE

	def _validate_inputs(self) -> None:
		if len(self.data) == 0:
			raise ValueError("data must contain at least one point cloud.")
		if len(self.data) != len(self.translations):
			raise ValueError("data and translations must have the same length.")
		if self.image_size[0] <= 0 or self.image_size[1] <= 0:
			raise ValueError("image_size must contain positive integers.")
		if self.point_size <= 0:
			raise ValueError("point_size must be greater than zero.")

		for idx, transform in enumerate(self.translations):
			matrix = np.asarray(transform)
			if matrix.shape != (4, 4):
				raise ValueError(
					f"translations[{idx}] must be a 4x4 transformation matrix, got {matrix.shape}."
				)

	def _build_point_clouds(self, o3d: Any) -> List[dict[str, Any]]:
		point_clouds: List[dict[str, Any]] = []

		for idx, cloud in enumerate(self.data):
			points = self._extract_points(cloud)
			if points.size == 0:
				continue

			color_bgr = np.asarray(self._PALETTE[idx % len(self._PALETTE)], dtype=np.uint8)
			color_rgb = color_bgr[::-1].astype(np.float64) / 255.0

			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
			pcd.paint_uniform_color(color_rgb.tolist())

			point_clouds.append(
				{
					"points": points,
					"geometry": pcd,
					"color_bgr": color_bgr,
				}
			)

		if len(point_clouds) == 0:
			raise ValueError("No valid points were found in data.")

		return point_clouds

	def _extract_points(self, cloud: Any) -> np.ndarray:
		if hasattr(cloud, "points"):
			points = np.asarray(cloud.points)
		else:
			points = np.asarray(cloud)

		if points.ndim != 2 or points.shape[1] < 3:
			raise ValueError(
				"Each point cloud must be an array-like object with shape (N, 3) or an Open3D point cloud."
			)

		return np.asarray(points[:, :3], dtype=np.float32)

	def _render_with_open3d(self, point_clouds: List[dict[str, Any]], o3d: Any) -> np.ndarray:
		width, height = self.image_size
		renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

		try:
			scene = renderer.scene
			bg = np.array(
				[
					self.background_color[2] / 255.0,
					self.background_color[1] / 255.0,
					self.background_color[0] / 255.0,
					1.0,
				],
				dtype=np.float32,
			)
			scene.set_background(bg)

			material = o3d.visualization.rendering.MaterialRecord()
			material.shader = "defaultUnlit"
			material.point_size = float(self.point_size)

			stacked = np.concatenate([item["points"] for item in point_clouds], axis=0)
			mins = stacked.min(axis=0)
			maxs = stacked.max(axis=0)
			center = (mins + maxs) / 2.0
			extent = np.maximum(maxs - mins, 1e-3)
			distance = float(max(extent[0], extent[1], extent[2]) * 2.5)
			eye = center + np.array([0.0, 0.0, distance], dtype=np.float32)
			up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

			for idx, item in enumerate(point_clouds):
				scene.add_geometry(f"cloud_{idx}", item["geometry"], material)

			renderer.setup_camera(60.0, center.tolist(), eye.tolist(), up.tolist())
			rendered = np.asarray(renderer.render_to_image())
		finally:
			renderer.scene.clear_geometry()
			del renderer

		if rendered.ndim != 3 or rendered.shape[2] < 3:
			raise ValueError("Failed to render a valid Open3D image.")

		return cv2.cvtColor(rendered[:, :, :3], cv2.COLOR_RGB2BGR)

	def _render_top_view(self, point_clouds: List[dict[str, Any]]) -> np.ndarray:
		width, height = self.image_size
		canvas = np.full((height, width, 3), self.background_color, dtype=np.uint8)

		stacked = np.concatenate([item["points"] for item in point_clouds], axis=0)
		xy = stacked[:, :2]
		xy_min = xy.min(axis=0)
		xy_max = xy.max(axis=0)
		xy_center = (xy_min + xy_max) / 2.0
		xy_extent = np.maximum(xy_max - xy_min, 1e-6)

		scale = 0.9 * min(width, height) / float(max(xy_extent[0], xy_extent[1]))
		radius = max(1, int(round(self.point_size / 2)))

		for item in point_clouds:
			points = item["points"]
			color = tuple(int(v) for v in item["color_bgr"].tolist())
			order = np.argsort(points[:, 2])
			xy_points = points[order, :2]

			px = np.round((xy_points[:, 0] - xy_center[0]) * scale + width / 2.0).astype(np.int32)
			py = np.round((xy_center[1] - xy_points[:, 1]) * scale + height / 2.0).astype(np.int32)

			valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
			for x, y in zip(px[valid], py[valid]):
				cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

		return canvas
