"""Render and project official H* equirectangular panoramas (PRE-06)."""

from __future__ import annotations

import math

import numpy as np


def _rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Return the camera-to-world rotation used by the official H* renderer."""
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    rotate_y = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ]
    )
    return rotate_y @ rotate_x


def equirectangular_to_perspective(
    panorama_rgb: np.ndarray,
    *,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one clean perspective RGB image from an H* panorama.

    The equation matches the official H* ``_e2p`` implementation, except that
    this function deliberately omits its green center crosshair.

    Args:
        panorama_rgb: Equirectangular RGB uint8 image ``[H,W,3]``.
        horizontal_fov_degrees: Horizontal pinhole field of view in degrees.
        yaw_degrees: View-center yaw in the official H* convention.
        pitch_degrees: View-center pitch in the official H* convention.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Perspective RGB uint8 image ``[height,width,3]``.

    Raises:
        ValueError: An image or camera parameter is invalid.
    """
    if (
        panorama_rgb.ndim != 3
        or panorama_rgb.shape[2] != 3
        or panorama_rgb.dtype != np.uint8
    ):
        raise ValueError("panorama_rgb must be uint8 [H,W,3]")
    if not 1.0 < horizontal_fov_degrees < 179.0:
        raise ValueError("horizontal FOV must be in (1,179) degrees")
    if not -90.0 <= pitch_degrees <= 90.0 or width <= 0 or height <= 0:
        raise ValueError("invalid perspective camera parameters")

    field = math.radians(horizontal_fov_degrees)
    focal = width / (2.0 * math.tan(field / 2.0))
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    camera = np.stack(
        (
            (pixel_x - width / 2.0) / focal,
            (pixel_y - height / 2.0) / focal,
            np.ones_like(pixel_x),
        ),
        axis=-1,
    )
    camera /= np.linalg.norm(camera, axis=-1, keepdims=True)
    world = camera.reshape(-1, 3) @ _rotation(yaw_degrees, pitch_degrees).T
    longitude = np.arctan2(world[:, 0], world[:, 2])
    latitude = np.arcsin(np.clip(world[:, 1], -1.0, 1.0))
    source_height, source_width = panorama_rgb.shape[:2]
    map_x = ((longitude / np.pi + 1.0) * 0.5 * source_width).reshape(height, width)
    map_y = ((latitude / np.pi + 0.5) * source_height).reshape(height, width)
    map_x %= source_width
    map_y = np.clip(map_y, 0.0, source_height - 1.0)
    x0 = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    x1 = (x0 + 1) % source_width
    y1 = np.minimum(y0 + 1, source_height - 1)
    weight_x = (map_x - x0)[..., None]
    weight_y = (map_y - y0)[..., None]
    top = panorama_rgb[y0, x0] * (1.0 - weight_x) + panorama_rgb[y0, x1] * weight_x
    bottom = panorama_rgb[y1, x0] * (1.0 - weight_x) + panorama_rgb[y1, x1] * weight_x
    return np.clip(top * (1.0 - weight_y) + bottom * weight_y, 0, 255).astype(np.uint8)


def project_spherical_points(
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray,
    *,
    view_yaw_degrees: float,
    view_pitch_degrees: float,
    horizontal_fov_degrees: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project official H* yaw/pitch points to perspective pixels.

    Args:
        yaw_degrees: Target yaws with arbitrary common shape.
        pitch_degrees: Target pitches with the same shape.
        view_yaw_degrees: Perspective view-center yaw.
        view_pitch_degrees: Perspective view-center pitch.
        horizontal_fov_degrees: Horizontal field of view.
        width: Output width.
        height: Output height.

    Returns:
        Pixel coordinates ``[...,2]``. Points behind the camera are NaN.
    """
    yaw = np.asarray(yaw_degrees, dtype=np.float64)
    pitch = np.asarray(pitch_degrees, dtype=np.float64)
    if yaw.shape != pitch.shape:
        raise ValueError("yaw and pitch arrays must have the same shape")
    longitude = np.radians(yaw)
    latitude = np.radians(-pitch)
    world = np.stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        ),
        axis=-1,
    )
    camera = world @ _rotation(view_yaw_degrees, view_pitch_degrees)
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    valid = camera[..., 2] > 0.0
    projected = np.full((*yaw.shape, 2), np.nan, dtype=np.float64)
    projected[..., 0][valid] = (
        focal * camera[..., 0][valid] / camera[..., 2][valid] + width / 2.0
    )
    projected[..., 1][valid] = (
        focal * camera[..., 1][valid] / camera[..., 2][valid] + height / 2.0
    )
    return projected
