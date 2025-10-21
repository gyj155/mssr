import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
import io
import base64
from typing import List, Tuple, Dict, Optional, Any


class VisualizationUtils:
    """
    A class for rendering point clouds from different camera viewpoints and overlaying arrows.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        
    def filter_point_cloud_by_height(self, world_points: np.ndarray, ground_normal: np.ndarray, 
                                   ground_centroid: np.ndarray, height_percentile_cutoff: float = 0.8,
                                   colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter point cloud by removing the top 20% highest points to avoid ceiling occlusion.
        """
        original_shape = world_points.shape
        
        world_points_flat = world_points.reshape(-1, 3)
            
        valid_mask = np.isfinite(world_points_flat).all(axis=1)
        valid_points = world_points_flat[valid_mask]

        ground_normal = np.array(ground_normal)
        ground_normal = ground_normal / np.linalg.norm(ground_normal)
        ground_centroid = np.array(ground_centroid)
        
        height_vectors = valid_points - ground_centroid
        heights = np.dot(height_vectors, ground_normal)
        
        min_height = np.min(heights)
        max_height = np.max(heights)
        height_range = max_height - min_height
        
        cutoff_height = min_height + height_range * height_percentile_cutoff
        
        height_mask = heights <= cutoff_height
        filtered_points = valid_points[height_mask]
        
        filtered_colors = None
        colors_flat = colors.reshape(-1, 3) if len(colors.shape) > 2 else colors
        colors_valid = colors_flat[valid_mask]
        filtered_colors = colors_valid[height_mask]
        return filtered_points, filtered_colors
    
    def calculate_elevated_camera_pose(
        self,
        original_extrinsic: np.ndarray,
        target_point: np.ndarray,
        ground_normal: np.ndarray,
        elevation_angle_deg: float = 45.0,
        assume_world_to_camera: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """
        With target_point as the center, perform pitch rotation (increase elevation angle)
        along the plane spanned by (view_dir, ground_normal), return the new
        world→camera extrinsic matrix (4x4).
        """

        extrinsic = np.asarray(original_extrinsic, dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        if extrinsic.shape != (4, 4):
            raise ValueError("original_extrinsic must be 3x4 or 4x4")

        target_point = np.asarray(target_point, dtype=float).reshape(3)
        ground_normal = np.asarray(ground_normal, dtype=float).reshape(3)
        gn_norm = np.linalg.norm(ground_normal)
        if gn_norm < 1e-12:
            raise ValueError("ground_normal has near-zero norm")
        ground_normal = ground_normal / gn_norm

        # ---- Extract camera pose from extrinsic (assuming world→camera) ----
        R_orig = extrinsic[:3, :3]
        t_orig = extrinsic[:3, 3]

        if not assume_world_to_camera:
            # [R|t]_cw  =>  world_to_camera = [R^T | -R^T t]
            R_w2c = R_orig.T
            t_w2c = -R_orig.T @ t_orig
            R_orig, t_orig = R_w2c, t_w2c

        # C = -R^T t
        camera_pos_world = -R_orig.T @ t_orig

        view_dir = target_point - camera_pos_world
        vd_norm = np.linalg.norm(view_dir)
        if vd_norm < 1e-12:
            raise ValueError("camera position equals target_point")
        view_dir = view_dir / vd_norm

        # Pitch rotation axis: perpendicular to both view_dir and ground_normal (rotation within the plane they span)
        rotation_axis = np.cross(ground_normal, view_dir)
        ax_norm = np.linalg.norm(rotation_axis)
        if ax_norm < 1e-8:
            # View direction parallel to ground normal: choose any axis orthogonal to ground_normal
            # we take a basis vector not parallel to ground_normal and orthogonalize it
            guess = np.array([1.0, 0.0, 0.0]) if abs(ground_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            rotation_axis = np.cross(ground_normal, guess)
            ax_norm = np.linalg.norm(rotation_axis)
            if ax_norm < 1e-8:
                raise ValueError("Failed to construct a valid rotation axis")
        rotation_axis /= ax_norm

        # Rotate camera position around this axis, with target as center
        angle = np.deg2rad(elevation_angle_deg)
        rot = R.from_rotvec(angle * rotation_axis)

        relative_pos = camera_pos_world - target_point
        new_relative_pos = rot.apply(relative_pos)
        new_camera_pos = target_point + new_relative_pos

        new_z = target_point - new_camera_pos
        new_z_norm = np.linalg.norm(new_z)
        if new_z_norm < 1e-12:
            raise ValueError("Degenerate new camera position")
        new_z = new_z / new_z_norm  # z-axis = forward (pointing to target)

        x = np.cross(new_z, ground_normal)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-8:
            ref = np.array([1.0, 0.0, 0.0]) if abs(new_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(ref, new_z)
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-8:
                raise ValueError("Failed to construct camera right axis")
        x /= x_norm

        y = np.cross(new_z, x)
        y /= np.linalg.norm(y)

        if np.dot(y, ground_normal) > 0:
            y = -y
            x = np.cross(y, new_z)
            x /= np.linalg.norm(x)

        new_R = np.vstack([x, y, new_z])

        new_t = -new_R @ new_camera_pos

        new_extrinsic = np.eye(4)
        new_extrinsic[:3, :3] = new_R
        new_extrinsic[:3, 3] = new_t

        if verbose:
            print(f"Camera elevated by {elevation_angle_deg}°")
            print(f"Original C: {camera_pos_world}")
            print(f"New C     : {new_camera_pos}")

        return new_extrinsic

    def render_point_cloud_to_image(self, world_points: np.ndarray, extrinsic: np.ndarray, 
                                intrinsic: np.ndarray, image_size: Tuple[int, int],
                                background_color: Tuple[int, int, int] = (255, 255, 255),
                                point_size: int = 2,
                                pre_extracted_colors: Optional[np.ndarray] = None,
                                arrows_info: Optional[Dict] = None) -> np.ndarray:

        width, height = image_size

        padding_info = None
        if arrows_info is not None:
            (new_width, new_height), (pad_left, pad_top, pad_right, pad_bottom) = self._calculate_required_size_with_arrows(
                image_size, extrinsic, intrinsic, arrows_info
            )
            width, height = new_width, new_height
            padding_info = (pad_left, pad_top, pad_right, pad_bottom)

        world_points_flat = world_points.reshape(-1, 3)

        
        if len(world_points_flat) == 0:
            return np.full((height, width, 3), background_color, dtype=np.uint8)
        
        extrinsic = np.array(extrinsic, dtype=np.float32)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        intrinsic = np.array(intrinsic, dtype=np.float32)
        
        world_points_h = np.column_stack([world_points_flat, np.ones(len(world_points_flat), dtype=np.float32)])
        camera_points_h = world_points_h @ extrinsic.T
        camera_points = camera_points_h[:, :3]
        
        front_mask = camera_points[:, 2] > 0
        if not np.any(front_mask):
            return np.full((height, width, 3), background_color, dtype=np.uint8)
            
        camera_points = camera_points[front_mask]
        
        projected = camera_points @ intrinsic.T
        pixel_coords = projected[:, :2] / projected[:, [2, 2]]
        
        if padding_info is not None:
            pad_left, pad_top, pad_right, pad_bottom = padding_info
            pixel_coords[:, 0] += pad_left
            pixel_coords[:, 1] += pad_top
        
        valid_mask = ((pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & 
                     (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height))
        
        if not np.any(valid_mask):
            return np.full((height, width, 3), background_color, dtype=np.uint8)
            
        pixel_coords = pixel_coords[valid_mask]
        depths = camera_points[valid_mask, 2]
        
        colors_flat = pre_extracted_colors.reshape(-1, 3)
        colors = colors_flat[front_mask][valid_mask]  
        image = self._render_points(
            pixel_coords, colors, depths, (width, height), 
            background_color, point_size
        )
        
        return image
    
    def extract_colors_from_images(self, world_points: np.ndarray, source_images: List) -> np.ndarray:
        """Extract colors for each point from corresponding source images."""
        original_shape = world_points.shape
        source_images_np = []
        for img in source_images:
                source_images_np.append(np.array(img))
        images_array = np.stack(source_images_np, axis=0)
        colors = images_array.reshape(-1, 3)
        return colors.reshape(original_shape)
    
    
    def _render_points(self, pixel_coords: np.ndarray, colors: np.ndarray, 
                               depths: np.ndarray, image_size: Tuple[int, int],
                               background_color: Tuple[int, int, int], 
                               point_size: int) -> np.ndarray:
        width, height = image_size
        image = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        pixel_coords_int = pixel_coords.astype(np.int32)
        
        depth_order = np.argsort(depths)[::-1]
        batch_size = min(1000, len(pixel_coords_int))
        
        for i in range(0, len(depth_order), batch_size):
            batch_indices = depth_order[i:i + batch_size]
            batch_coords = pixel_coords_int[batch_indices]
            batch_colors = colors[batch_indices]
            
            for j, (u, v) in enumerate(batch_coords):
                color = batch_colors[j]
                cv2.circle(image, (u, v), point_size, 
                            (int(color[2]), int(color[1]), int(color[0])), -1)
        
        return image
    
    def world_to_camera_coordinates(self, world_point: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """Convert world coordinates to camera coordinates using extrinsic matrix"""
        if len(world_point) == 3:
            world_point_h = np.append(world_point, 1)
        else:
            world_point_h = world_point
        camera_point_h = extrinsic @ world_point_h
        return camera_point_h[:3]
    
    def world_direction_to_camera(self, world_direction: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """Convert world direction vector to camera coordinates (only rotation, no translation)"""
        rotation_matrix = extrinsic[:3, :3]
        camera_direction = rotation_matrix @ world_direction
        return camera_direction
    
    def camera_3d_to_pixel(self, point_3d: np.ndarray, intrinsic: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D camera coordinates to pixel coordinates"""
        if point_3d[2] <= 0:
            return None
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x_norm = point_3d[0] / point_3d[2]
        y_norm = point_3d[1] / point_3d[2]
        u = fx * x_norm + cx
        v = fy * y_norm + cy
        return np.array([u, v])
    
    def draw_arrow_3d_on_image(self, img: np.ndarray, origin_3d: np.ndarray, direction_3d: np.ndarray, 
                            arrow_length_3d: float, color: Tuple[int, int, int], intrinsic: np.ndarray,
                            line_thickness: int = 3, arrow_tip_ratio: float = 0.15, 
                            offset_2d: Tuple[int, int] = (0, 0)) -> bool:
        """Draw 3D arrow projected to 2D image (reusing logic from original module)"""
        endpoint_3d = origin_3d + direction_3d * arrow_length_3d
        origin_pixel = self.camera_3d_to_pixel(origin_3d, intrinsic)
        endpoint_pixel = self.camera_3d_to_pixel(endpoint_3d, intrinsic)
        
        if origin_pixel is None or endpoint_pixel is None:
            return False
        
        offset_arr = np.array(offset_2d)
        origin_pixel += offset_arr
        endpoint_pixel += offset_arr
        
        h, w = img.shape[:2]
        
        cv2.line(img, tuple(origin_pixel.astype(int)), tuple(endpoint_pixel.astype(int)), 
                color, line_thickness)
        
        direction_2d = endpoint_pixel - origin_pixel
        length_2d = np.linalg.norm(direction_2d)
        
        if length_2d > 0:
            unit_direction_2d = direction_2d / length_2d
            tip_length = length_2d * arrow_tip_ratio
            perpendicular = np.array([-unit_direction_2d[1], unit_direction_2d[0]])
            
            tip_point1 = endpoint_pixel - tip_length * unit_direction_2d + tip_length * 0.5 * perpendicular
            tip_point2 = endpoint_pixel - tip_length * unit_direction_2d - tip_length * 0.5 * perpendicular
            
            cv2.line(img, tuple(endpoint_pixel.astype(int)), tuple(tip_point1.astype(int)), 
                    color, line_thickness)
            cv2.line(img, tuple(endpoint_pixel.astype(int)), tuple(tip_point2.astype(int)), 
                    color, line_thickness)
        
        return True
    
    def add_text_label_3d_on_image(self, img: np.ndarray, origin_3d: np.ndarray, direction_3d: np.ndarray, 
                                arrow_length_3d: float, text: str, color: Tuple[int, int, int], 
                                intrinsic: np.ndarray, offset_2d: Tuple[int, int] = (0, 0),
                                font_scale: float = 0.7, thickness: int = 2, background_padding: int = 2) -> Optional[Dict]:
        """Add text label near the end of 3D arrow (reusing logic from original module)"""
        endpoint_3d = origin_3d + direction_3d * arrow_length_3d
        origin_pixel = self.camera_3d_to_pixel(origin_3d, intrinsic)
        endpoint_pixel = self.camera_3d_to_pixel(endpoint_3d, intrinsic)
        
        if origin_pixel is None or endpoint_pixel is None:
            return None
        
        offset_arr = np.array(offset_2d)
        origin_pixel += offset_arr
        endpoint_pixel += offset_arr
        
        arrow_direction_2d = endpoint_pixel - origin_pixel
        arrow_length_2d = np.linalg.norm(arrow_direction_2d)
        
        if arrow_length_2d > 0:
            unit_direction_2d = arrow_direction_2d / arrow_length_2d
            label_offset = 25
            label_pixel = endpoint_pixel + unit_direction_2d * label_offset
        else:
            label_pixel = endpoint_pixel
        
        h, w = img.shape[:2]
        if not (-50 <= label_pixel[0] <= w + 50 and -50 <= label_pixel[1] <= h + 50):
            return None
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x = int(label_pixel[0] - text_width // 2)
        text_y = int(label_pixel[1] + text_height // 2)
        
        # Draw white background
        cv2.rectangle(img, 
                    (text_x - background_padding, text_y - text_height - background_padding),
                    (text_x + text_width + background_padding, text_y + baseline + background_padding),
                    (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return {
            'top_left': (text_x - background_padding, text_y - text_height - background_padding),
            'bottom_right': (text_x + text_width + background_padding, text_y + baseline + background_padding)
        }
    
    def overlay_arrows_on_rendered_image(self, rendered_image: np.ndarray, object_position: np.ndarray,
                                    directions_world: Dict[str, np.ndarray], extrinsic: np.ndarray,
                                    intrinsic: np.ndarray, arrow_length_3d: float = 0.4,
                                    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
                                    font_scale: float = 0.7, offset_2d: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Overlay directional arrows on rendered image.
        """
        img = rendered_image.copy()
        
        if colors is None:
            colors = {
                '0': (0, 0, 255),      # Red
                '1': (255, 0, 0),      # Blue  
                '2': (0, 180, 200),    # Yellow
                '3': (0, 180, 0),      # Green
                '4': (150, 150, 0),    # Teal
            }
        
        extrinsic_matrix = np.array(extrinsic)
        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
        
        world_origin = np.array(object_position, dtype=np.float64)
        origin_3d_camera = self.world_to_camera_coordinates(world_origin, extrinsic_matrix)
        
        if origin_3d_camera[2] <= 0:
            print(f"Warning: Object is behind camera in rendered view (depth: {origin_3d_camera[2]:.3f})")
            return img
        
        directions_camera = {}
        for direction_name, world_direction in directions_world.items():
            camera_direction = self.world_direction_to_camera(world_direction, extrinsic_matrix)
            directions_camera[direction_name] = camera_direction
        
        intrinsic_np = np.array(intrinsic)
        
        for direction_name in sorted(directions_world.keys()):
            direction_3d_camera = directions_camera[direction_name]
            color = colors.get(direction_name, (128, 128, 128))
            
            success = self.draw_arrow_3d_on_image(
                img, origin_3d_camera, direction_3d_camera, arrow_length_3d, 
                color, intrinsic_np, line_thickness=3, offset_2d=offset_2d
            )
            
            if success:
                self.add_text_label_3d_on_image(
                    img, origin_3d_camera, direction_3d_camera, arrow_length_3d, 
                    direction_name, color, intrinsic_np, font_scale=font_scale, thickness=2, offset_2d=offset_2d
                )
        
        origin_pixel = self.camera_3d_to_pixel(origin_3d_camera, intrinsic_np)
        if origin_pixel is not None:
            offset_arr = np.array(offset_2d)
            origin_pixel_with_offset = origin_pixel + offset_arr
            cv2.circle(img, (int(origin_pixel_with_offset[0]), int(origin_pixel_with_offset[1])), 5, (0, 0, 0), -1)
        
        return img
    
    def concatenate_images_horizontally(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Concatenate two images horizontally, handling different heights.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        max_height = max(h1, h2)
        
        if h1 < max_height:
            pad_top = (max_height - h1) // 2
            pad_bottom = max_height - h1 - pad_top
            img1 = cv2.copyMakeBorder(img1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        if h2 < max_height:
            pad_top = (max_height - h2) // 2
            pad_bottom = max_height - h2 - pad_top
            img2 = cv2.copyMakeBorder(img2, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        concatenated = np.hstack([img1, img2])
        return concatenated
    
    def _calculate_required_size_with_arrows(self, original_size: Tuple[int, int], 
                                           extrinsic: np.ndarray, intrinsic: np.ndarray,
                                           arrows_info: Dict) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        Calculate required image size and padding to accommodate arrows and labels.
        """
        width, height = original_size
        
        origin_world = np.array(arrows_info['origin'])
        directions_world = arrows_info['directions']
        arrow_length_3d = arrows_info.get('arrow_length', 0.4)
        font_scale = arrows_info.get('font_scale', 0.7)
        
        extrinsic_matrix = np.array(extrinsic)
        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
        
        origin_3d_camera = self.world_to_camera_coordinates(origin_world, extrinsic_matrix)
        
        if origin_3d_camera[2] <= 0:
            return original_size
        
        all_points_2d = []
        
        origin_pixel = self.camera_3d_to_pixel(origin_3d_camera, intrinsic)
        if origin_pixel is not None:
            all_points_2d.append(origin_pixel)
        
        arrow_tip_ratio = 0.15
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        background_padding = 2
        
        for direction_name, world_direction in directions_world.items():
            camera_direction = self.world_direction_to_camera(world_direction, extrinsic_matrix)
            
            endpoint_3d_camera = origin_3d_camera + camera_direction * arrow_length_3d
            endpoint_pixel = self.camera_3d_to_pixel(endpoint_3d_camera, intrinsic)
            
            if origin_pixel is None or endpoint_pixel is None:
                continue
            
            all_points_2d.append(endpoint_pixel)
            
            direction_2d = endpoint_pixel - origin_pixel
            length_2d = np.linalg.norm(direction_2d)
            
            if length_2d > 0:
                unit_direction_2d = direction_2d / length_2d
                tip_length = length_2d * arrow_tip_ratio
                perpendicular = np.array([-unit_direction_2d[1], unit_direction_2d[0]])
                
                tip_point1 = endpoint_pixel - tip_length * unit_direction_2d + tip_length * 0.5 * perpendicular
                tip_point2 = endpoint_pixel - tip_length * unit_direction_2d - tip_length * 0.5 * perpendicular
                all_points_2d.extend([tip_point1, tip_point2])
                
                label_offset = 25
                label_pixel = endpoint_pixel + unit_direction_2d * label_offset
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    str(direction_name), font, font_scale, thickness
                )
                
                text_x = label_pixel[0] - text_width // 2
                text_y = label_pixel[1] + text_height // 2
                
                label_top_left = np.array([text_x - background_padding, 
                                         text_y - text_height - background_padding])
                label_bottom_right = np.array([text_x + text_width + background_padding, 
                                             text_y + baseline + background_padding])
                all_points_2d.extend([label_top_left, label_bottom_right])
                
        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
        
        if all_points_2d:
            points_arr = np.array(all_points_2d)
            min_x, min_y = np.min(points_arr, axis=0)
            max_x, max_y = np.max(points_arr, axis=0)
        
            pad_left = int(max(0, -min_x))
            pad_top = int(max(0, -min_y))
            pad_right = int(max(0, max_x - width))
            pad_bottom = int(max(0, max_y - height))
            
            new_width = width + pad_left + pad_right
            new_height = height + pad_top + pad_bottom

            return (new_width, new_height), (pad_left, pad_top, pad_right, pad_bottom)
        
        return original_size, (0, 0, 0, 0)


def html_embed_image_from_array(image_array: np.ndarray, size: int = None) -> str:
    """
    Convert numpy image array to HTML embedded image string.
    """
    pil_image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    if size:
        return f'<img src="data:image/png;base64,{base64_image}" style="max-width: {size}px;">'
    else:
        return f'<img src="data:image/png;base64,{base64_image}">'
