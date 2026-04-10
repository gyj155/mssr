import re
import torch
import numpy as np
import io
import json
import os
import base64
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import groundingdino.datasets.transforms as T
from src.prompts.vqa_prompt import (
    FIND_OBJ_PROMPT_MMSI,
)
from .engine_utils import *
from groundingdino.util.inference import load_model, predict
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import matplotlib.pyplot as plt
from src.prompts.SOG_prompt import COARSE_PROMPT, FINE_PROMPT, IDENTIFY_KEY_OBJECT_PROMPT, UNIFIED_OBJECT_SELECTION_PROMPT
from .visualization_utils import VisualizationUtils

class PredefinedModule:
    def __init__(self, name, trace_path=None):
        self.trace_path = trace_path
        self.name = name

    def write_trace(self, html):
        if self.trace_path:
            with open(self.trace_path, "a+", encoding="utf-8") as f:
                f.write(f"{html}\n")

class LocateModule(PredefinedModule):
    def __init__(
        self,
        dataset,
        grounding_dino=None,
        trace_path=None,
    ):
        super().__init__("loc", trace_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        self.grounding_dino = grounding_dino
        self.BOX_THRESHOLD = 0.25
        self.TEXT_THRESHOLD = 0.25


    def _parse_bounding_boxes(self, boxes, width, height):
        if len(boxes) == 0:
            return []

        bboxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            bboxes.append(
                [
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                ]
            )
        return bboxes

    def transform_image(self, og_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        og_image = og_image.convert("RGB")
        img = np.asarray(og_image)
        im_t, _ = transform(og_image, None)
        return img, im_t

    def locate_bboxs(self, image, object_prompt):
        original_object_prompt = object_prompt
        width, height = image.size
        prompt = f"{object_prompt.replace(' ', '-')} ."
        _, img_gd = self.transform_image(image)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=img_gd,
                caption=prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
            )
        bboxes = self._parse_bounding_boxes(boxes, width, height)

        if len(bboxes) == 0:
            self.write_trace(f"<p> No objects found<p>")
            return []

        self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        boxed_image = box_image(image, bboxes)
        boxed_html = html_embed_image(boxed_image)
        self.write_trace(boxed_html)
        if len(bboxes) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(bboxes)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Boxes: {bboxes}, return the first box<p>")

        return bboxes[0]

class FindObjModule(PredefinedModule):
    def __init__(
        self,
        trace_path=None,
        modules_list=None,
        vqa_model="gemini-3.1-flash-lite-preview",
    ):
        super().__init__("find_obj", trace_path)
        self.generator = Generator(vqa_model, temperature=1.0)

    def find(self, images, object_prompts):   
        image_ids_str = ", ".join(map(str, range(len(images))))
        object_prompts_str = ",".join(object_prompts)
        prompt = FIND_OBJ_PROMPT_MMSI.format(object_prompts=object_prompts_str, image_ids=image_ids_str)
        
        content = [{"type": "text", "text": prompt}]
        
        self.write_trace(f"<h2>Finding objects: {', '.join(object_prompts)}</h2>")
        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            })
            self.write_trace(f"<p>Image {i}:</p>")
            self.write_trace(html_embed_image(image, 128))

        messages = [{"role": "user", "content": content}]
        
        output, _ = self.generator.generate("", messages)

        self.write_trace(f"<p><b>Find Object Prompt:</b> {prompt}</p>")
        
        
        try:
            answer_json_str = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].strip()
            answer_dict = json.loads(answer_json_str)
            self.write_trace(f"<p><b>Found objects: {answer_dict}</b></p>")
            return answer_dict
        except (IndexError, json.JSONDecodeError) as e:
            self.write_trace(f"<p><b>Error parsing model output: {e}. Output was: {output}</b></p>")
            return {}

class RelativeCamMovementModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("relative_cam_movement", trace_path)

    def calculate(self, cam_extrinsic0, cam_extrinsic1, z_is_positive_forward=True):
        """
        Calculates the relative movement of camera1 with respect to camera0's coordinate system.
        """
        self.write_trace(f"<p>Relative Camera Movement</p>")

        E0 = np.array(cam_extrinsic0)
        E1 = np.array(cam_extrinsic1)

        # Check if matrices need to be augmented from 3x4 to 4x4
        if E0.shape == (3, 4):
            E0 = np.vstack([E0, [0, 0, 0, 1]])
        if E1.shape == (3, 4):
            E1 = np.vstack([E1, [0, 0, 0, 1]])

        if E0.shape != (4, 4) or E1.shape != (4, 4):
            raise ValueError(
                f"Extrinsic matrices must be 4x4 or 3x4. Got {E0.shape} and {E1.shape}"
            )

        # The transformation from world to cam0 is E0. The transformation from world to cam1 is E1.
        # The pose of cam0 in the world is T_w_c0 = inv(E0).
        # The pose of cam1 in the world is T_w_c1 = inv(E1).
        # The pose of cam1 relative to cam0 is T_c0_c1 = T_c0_w @ T_w_c1 = E0 @ np.linalg.inv(E1).
        T_c0_c1 = E0 @ np.linalg.inv(E1)
        rotation_matrix = T_c0_c1[:3, :3]
        translation_vector = T_c0_c1[:3, 3]

        # Decompose rotation matrix into Euler angles (yaw, pitch, roll) in degrees
        # Using 'yxz' order: yaw (around y), pitch (around x), roll (around z)
        # This order is often intuitive for camera orientation.
        angles = R.from_matrix(rotation_matrix).as_euler("yxz", degrees=True)
        yaw, pitch, roll = angles[0], angles[1], angles[2]
        # +X right, +Y down
        tx, ty, tz = translation_vector[0], translation_vector[1], translation_vector[2]
        forward = tz
        up = -ty
        rotate_right = yaw
        rotate_up = pitch
        rotate_roll = roll

        if z_is_positive_forward:
            around_x = pitch
            around_y = yaw
            around_z = roll
        else:
            around_x = pitch
            around_y = -yaw
            around_z = -roll

        result = {
            "forward": float(forward),
            "right": float(tx),
            "up": float(up),
            "rotate_right": float(rotate_right),
            "rotate_up": float(rotate_up),
            "rotate_roll": float(rotate_roll),
            "around_x": float(around_x),
            "around_y": float(around_y),
            "around_z": float(around_z)
        }
        self.write_trace(f"<p>Result: {result}</p>")
        return result

class RelativeObjectPositionModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("relative_object_position", trace_path)

    def calculate(self, cam_extrinsic, object_position):
        """
        Calculates the relative position of an object with respect to the camera's coordinate system.
        """
        self.write_trace(f"<p>Relative Object Position</p>")
        E_w2c = np.array(cam_extrinsic)
        P_world = np.array(object_position)

        if E_w2c.shape == (3, 4):
            E_w2c = np.vstack([E_w2c, [0, 0, 0, 1]])
            
        if E_w2c.shape != (4, 4):
            raise ValueError(
                f"Extrinsic matrix must be 4x4 or 3x4. Got {E_w2c.shape}"
            )
        
        if P_world.shape != (3,):
            raise ValueError(
                f"Object position must be a 3-element vector. Got {P_world.shape}"
            )

        P_world_h = np.append(P_world, 1)
        P_cam_h = E_w2c @ P_world_h
        # Standard computer vision coordinate system: +X right, +Y down, +Z forward
        tx, ty, tz = P_cam_h[0], P_cam_h[1], P_cam_h[2]
        
        forward = tz
        right = tx
        up = -ty  # +Y is down in standard CV coordinates

        result = {
            "forward": float(forward),
            "right": float(right),
            "up": float(up),
        }
        self.write_trace(f"<p>Result: {result}</p>")
        return result

class GetGeoInfoModule(PredefinedModule):
    def __init__(self, vggt_model, device, trace_path=None):
        super().__init__("get_geo_info", trace_path)
        self.vggt_model = vggt_model
        self.device = device

    def extract(self, images):
        """
        Gets geometric information from a list of images in the same scene.
        Images should already be preprocessed to the correct size.
        """
        self.write_trace("<p>Getting geometric info...</p>")
        
        # Images are already preprocessed, just convert to tensor
        from torchvision.transforms import ToTensor
        to_tensor = ToTensor()
        image_tensors = []
        
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = to_tensor(img)
            image_tensors.append(img_tensor)
        
        # Stack tensors
        processed_images = torch.stack(image_tensors).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = self.vggt_model(processed_images)

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], processed_images.shape[-2:]
        )
        depth_maps = predictions["depth"]
        extrinsics_np = extrinsics.cpu().numpy().squeeze(0)
        intrinsics_np = intrinsics.cpu().numpy().squeeze(0)
        depth_maps_np = depth_maps.cpu().numpy().squeeze(0)#(N, W, H, 1)

        world_points = unproject_depth_map_to_point_map(depth_maps_np, extrinsics_np, intrinsics_np)
        world_points = world_points.reshape(-1, 3)

        
        self.write_trace(f"<p>Successfully extracted geometric info for {len(images)} images. Extrinsics: {extrinsics_np.shape}, Intrinsics: {intrinsics_np.shape}, Depth Maps: {depth_maps_np.shape}</p>")

        return extrinsics_np, intrinsics_np, depth_maps_np, world_points

class CalibrateDirectionsModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("calibrate_directions", trace_path)

    def calibrate(self, world_points, target_3d_point, center_3d_point, known_direction="west", ground_normal=None):
        """
        Calibrates the cardinal directions (North, South, East, West) based on known spatial relationship.
        """
        self.write_trace("<p>Calibrating directions...</p>")

        # 1. Use provided ground normal or find ground plane using PCA
        if ground_normal is not None:
            self.write_trace("<p>Using pre-computed ground plane normal</p>")
            ground_normal = np.array(ground_normal)
            # Calculate centroid for consistency
            points = np.array(world_points)
            if len(points.shape) == 3:  # If points is from multiple images
                points = points.reshape(-1, 3)
            centroid = points.mean(axis=0)
        else:
            self.write_trace("<p>Computing ground plane using PCA...</p>")
            points = np.array(world_points)
            if len(points.shape) == 3:  # If points is from multiple images
                points = points.reshape(-1, 3)
            
            centroid = points.mean(axis=0)
            centered_points = points - centroid
            cov_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            ground_normal = eigenvectors[:, np.argmin(eigenvalues)]
            if ground_normal[1] > 0:
            #Notice according to vggt setting, the world coordinate is the same as the first camera's extrinsic coordinate.(+Z forward, -Y upward). 
            #As we want normal vector points 'upward' in real world. The Y should be negative. 
                ground_normal *= -1
        
        self.write_trace(f"<p>Ground plane normal: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]</p>")
        
        # 2. Handle scene center case and project target and center points onto ground plane
        # If target or center is None, use the scene centroid (average of all points)
        if target_3d_point is None:
            target_point = centroid.copy()
            self.write_trace("<p>Using scene center for target point</p>")
        else:
            target_point = np.array(target_3d_point)
            
        if center_3d_point is None:
            center_point = centroid.copy()
            self.write_trace("<p>Using scene center for center point</p>")
        else:
            center_point = np.array(center_3d_point)
        
        # Project points to ground plane
        target_proj = target_point - np.dot(target_point - centroid, ground_normal) * ground_normal
        center_proj = center_point - np.dot(center_point - centroid, ground_normal) * ground_normal
        
        # 3. Calculate known direction vector
        known_vec = target_proj - center_proj
        
        # Check if target and center are the same (both None or very close)
        assert np.linalg.norm(known_vec) > 1e-6, "Target and center are at the same location. Cannot calibrate directions."
            
        known_vec_2d = known_vec - np.dot(known_vec, ground_normal) * ground_normal  # Ensure it's on plane
        known_vec_2d = known_vec_2d / np.linalg.norm(known_vec_2d)  # Normalize
        
        # 4. Calculate all cardinal directions based on known direction
        # Create rotation matrix for 90 degrees around ground normal
        def rotate_vector_on_plane(vec, normal, angle_deg):
            angle_rad = np.radians(angle_deg)
            # Rodrigues' rotation formula
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            return cos_angle * vec + sin_angle * np.cross(normal, vec) + \
                    (1 - cos_angle) * np.dot(vec, normal) * normal
        
        # Determine directions based on known direction
        known_dir_lower = known_direction.lower()
        
        if known_dir_lower == "west":
            west = known_vec_2d
            north = rotate_vector_on_plane(west, ground_normal, -90)
            east = -west
            south = -north
        elif known_dir_lower == "east":
            east = known_vec_2d
            north = rotate_vector_on_plane(east, ground_normal, 90)
            west = -east
            south = -north
        elif known_dir_lower == "north":
            north = known_vec_2d
            east = rotate_vector_on_plane(north, ground_normal, -90)
            south = -north
            west = -east
        elif known_dir_lower == "south":
            south = known_vec_2d
            east = rotate_vector_on_plane(south, ground_normal, 90)
            north = -south
            west = -east
        elif known_dir_lower == "northeast":
            # Northeast is 45 degrees between north and east
            northeast = known_vec_2d
            north = rotate_vector_on_plane(northeast, ground_normal, 45)
            east = rotate_vector_on_plane(northeast, ground_normal, -45)
            south = -north
            west = -east
        elif known_dir_lower == "northwest":
            # Northwest is 45 degrees between north and west
            northwest = known_vec_2d
            north = rotate_vector_on_plane(northwest, ground_normal, -45)
            west = rotate_vector_on_plane(northwest, ground_normal, 45)
            south = -north
            east = -west
        elif known_dir_lower == "southeast":
            # Southeast is 45 degrees between south and east
            southeast = known_vec_2d
            south = rotate_vector_on_plane(southeast, ground_normal, -45)
            east = rotate_vector_on_plane(southeast, ground_normal, 45)
            north = -south
            west = -east
        elif known_dir_lower == "southwest":
            # Southwest is 45 degrees between south and west
            southwest = known_vec_2d
            south = rotate_vector_on_plane(southwest, ground_normal, 45)
            west = rotate_vector_on_plane(southwest, ground_normal, -45)
            north = -south
            east = -west
        else:
            raise ValueError(f"Unknown direction: {known_direction}. Supported directions: north, south, east, west, northeast, northwest, southeast, southwest")
        
        # 5. Create visualization
        self._visualize_calibration(center_proj, target_proj, north, south, east, west, 
                                    ground_normal, known_direction, centroid)
        
        result = {
            "north": north,
            "south": south,
            "east": east,
            "west": west,
            "ground_normal": ground_normal,
            "ground_centroid": centroid
        }
        
        self.write_trace(f"<p>Calibration complete. Directions calibrated based on: {known_direction}</p>")
        return result

    def calibrate_from_vector(self, world_points, direction_vector, center_3d_point=None, known_direction="north", ground_normal=None):
        """
        Calibrates cardinal directions directly from a known direction vector.
        This is a simplified wrapper that calls the existing calibrate function.
        """
        self.write_trace("<p>Calibrating directions from direction vector...</p>")
        
        direction_vector_np = np.array(direction_vector)
        
        if center_3d_point is None:
            points = np.array(world_points)
            if len(points.shape) == 3:
                points = points.reshape(-1, 3)
            center_3d_point = points.mean(axis=0).tolist()
            self.write_trace("<p>Using scene center for center point</p>")
        
        center_3d_point_np = np.array(center_3d_point)
        hypothetical_target_point = (center_3d_point_np + direction_vector_np * 1.0).tolist()
        
        return self.calibrate(
            world_points=world_points,
            target_3d_point=hypothetical_target_point,
            center_3d_point=center_3d_point,
            known_direction=known_direction,
            ground_normal=ground_normal
        )
            
    
    def _visualize_calibration(self, center, target, north, south, east, west, normal, known_dir, centroid):
        """Creates a BEV visualization of the calibration."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Project points to 2D for visualization (remove component along normal)
        def project_to_2d(point):
            # Use the two largest components for visualization
            if abs(normal[2]) > 0.9:  # If normal is mostly vertical, use X-Y plane
                return point[0], point[1]
            else:  # Otherwise project properly
                # Create basis vectors on the plane
                v1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
                v1 = v1 - np.dot(v1, normal) * normal
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                return np.dot(point - centroid, v1), np.dot(point - centroid, v2)
        
        center_2d = project_to_2d(center)
        target_2d = project_to_2d(target)
        
        # Plot center and target
        ax.scatter(*center_2d, color='blue', s=200, marker='o', label='Center', zorder=5)
        ax.scatter(*target_2d, color='red', s=200, marker='*', label=f'Target ({known_dir})', zorder=5)
        
        # Draw arrow from center to target
        ax.annotate('', xy=target_2d, xytext=center_2d,
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        
        # Draw cardinal directions
        scale = np.linalg.norm(np.array(target_2d) - np.array(center_2d)) * 0.8
        directions = {
            'N': north * scale,
            'S': south * scale,
            'E': east * scale,
            'W': west * scale
        }
        
        colors = {'N': 'green', 'S': 'darkgreen', 'E': 'orange', 'W': 'darkorange'}
        for label, vec in directions.items():
            end_point = np.array(center) + vec
            end_2d = project_to_2d(end_point)
            ax.annotate('', xy=end_2d, xytext=center_2d,
                       arrowprops=dict(arrowstyle='->', color=colors[label], lw=2))
            ax.text(end_2d[0], end_2d[1], label, fontsize=14, fontweight='bold',
                   ha='center', va='center', color=colors[label])
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Direction Calibration (Bird\'s Eye View)', fontsize=16)
        ax.legend()
        
        # Save to bytes for HTML embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Embed in HTML
        img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        self.write_trace(f'<img src="data:image/png;base64,{img_base64}" style="max-width: 600px;">')

class CalculateDirectionModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("calculate_direction", trace_path)

    def calculate(self, calibration_info, target_3d_point, center_3d_point, ground_normal=None):
        """
        Calculates the direction of target relative to center using calibrated directions.
        """
        self.write_trace("<p>Calculating relative direction...</p>")

        # Extract calibration data
        north = np.array(calibration_info["north"])
        south = np.array(calibration_info["south"])
        east = np.array(calibration_info["east"])
        west = np.array(calibration_info["west"])
        
        # Use provided ground normal or get from calibration info
        if ground_normal is not None:
            self.write_trace("<p>Using pre-computed ground plane normal</p>")
            ground_normal = np.array(ground_normal)
        else:
            ground_normal = np.array(calibration_info["ground_normal"])
            
        ground_centroid = np.array(calibration_info["ground_centroid"])
        
        # Handle scene center case and project points to ground plane
        # If target or center is None, use the scene centroid (from calibration)
        if target_3d_point is None:
            target_point = ground_centroid.copy()
            self.write_trace("<p>Using scene center for target point</p>")
        else:
            target_point = np.array(target_3d_point)
            
        if center_3d_point is None:
            center_point = ground_centroid.copy()
            self.write_trace("<p>Using scene center for center point</p>")
        else:
            center_point = np.array(center_3d_point)
        
        target_proj = target_point - np.dot(target_point - ground_centroid, ground_normal) * ground_normal
        center_proj = center_point - np.dot(center_point - ground_centroid, ground_normal) * ground_normal
        
        # Calculate direction vector from center to target
        direction_vec = target_proj - center_proj
        direction_vec_2d = direction_vec - np.dot(direction_vec, ground_normal) * ground_normal
        
        if np.linalg.norm(direction_vec_2d) < 1e-6:
            self.write_trace("<p>Target and center are at the same location</p>")
            return "same location"
        
        direction_vec_2d = direction_vec_2d / np.linalg.norm(direction_vec_2d)
        
        # Calculate angles with cardinal directions
        angles = {
            "north": np.arccos(np.clip(np.dot(direction_vec_2d, north), -1, 1)),
            "south": np.arccos(np.clip(np.dot(direction_vec_2d, south), -1, 1)),
            "east": np.arccos(np.clip(np.dot(direction_vec_2d, east), -1, 1)),
            "west": np.arccos(np.clip(np.dot(direction_vec_2d, west), -1, 1))
        }
        
        angles_deg = {k: np.degrees(v) for k, v in angles.items()}
        
        primary_dir = min(angles_deg, key=angles_deg.get)
        primary_angle = angles_deg[primary_dir]
        
        # Determine if it's a compound direction (e.g., northeast)
        threshold = 22.5  # Half of 45 degrees
        if primary_angle < threshold:
            direction = primary_dir
        else:
            # Check for compound directions
            if angles_deg["north"] < 67.5 and angles_deg["east"] < 67.5:
                direction = "northeast"
            elif angles_deg["north"] < 67.5 and angles_deg["west"] < 67.5:
                direction = "northwest"
            elif angles_deg["south"] < 67.5 and angles_deg["east"] < 67.5:
                direction = "southeast"
            elif angles_deg["south"] < 67.5 and angles_deg["west"] < 67.5:
                direction = "southwest"
            else:
                direction = primary_dir
        
        # Create visualization
        self._visualize_direction(center_proj, target_proj, north, south, east, west, 
                                ground_normal, direction, ground_centroid)
        
        self.write_trace(f"<p>Direction: Target is to the <b>{direction}</b> of center</p>")
        self.write_trace(f"<p>Angles - N: {angles_deg['north']:.1f}°, S: {angles_deg['south']:.1f}°, "
                        f"E: {angles_deg['east']:.1f}°, W: {angles_deg['west']:.1f}°</p>")
        
        result = {
            "direction": direction,
            "angles_deg": angles_deg
        }
        
        return result
            
    
    def _visualize_direction(self, center, target, north, south, east, west, normal, direction, centroid):
        """Creates a BEV visualization showing the relative direction."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Project points to 2D for visualization
        def project_to_2d(point):
            if abs(normal[2]) > 0.9:
                return point[0], point[1]
            else:
                v1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
                v1 = v1 - np.dot(v1, normal) * normal
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                return np.dot(point - centroid, v1), np.dot(point - centroid, v2)
        
        center_2d = project_to_2d(center)
        target_2d = project_to_2d(target)
        
        # Draw compass rose at center
        scale = np.linalg.norm(np.array(target_2d) - np.array(center_2d)) * 0.4
        directions_compass = {
            'N': north * scale,
            'S': south * scale,
            'E': east * scale,
            'W': west * scale
        }
        
        for label, vec in directions_compass.items():
            end_point = np.array(center) + vec
            end_2d = project_to_2d(end_point)
            ax.plot([center_2d[0], end_2d[0]], [center_2d[1], end_2d[1]], 
                   color='gray', alpha=0.5, linestyle='--')
            ax.text(end_2d[0], end_2d[1], label, fontsize=12, 
                   ha='center', va='center', color='gray')
        
        # Highlight the detected direction
        direction_colors = {
            'north': 'green', 'south': 'darkgreen', 
            'east': 'orange', 'west': 'darkorange',
            'northeast': 'yellowgreen', 'northwest': 'olive',
            'southeast': 'coral', 'southwest': 'chocolate'
        }
        
        # Draw arrow from center to target
        ax.annotate('', xy=target_2d, xytext=center_2d,
                   arrowprops=dict(arrowstyle='->', 
                                 color=direction_colors.get(direction, 'purple'), 
                                 lw=3))
        
        # Plot points
        ax.scatter(*center_2d, color='blue', s=300, marker='o', label='Center', zorder=5)
        ax.scatter(*target_2d, color='red', s=300, marker='*', label='Target', zorder=5)
        
        # Add direction label
        mid_x = (center_2d[0] + target_2d[0]) / 2
        mid_y = (center_2d[1] + target_2d[1]) / 2
        ax.text(mid_x, mid_y, direction.upper(), fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Target is to the {direction.upper()} of Center', fontsize=16)
        ax.legend()
        
        # Save to bytes for HTML embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Embed in HTML
        img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        self.write_trace(f'<img src="data:image/png;base64,{img_base64}" style="max-width: 600px;">')

class GetObject3DPositionModule(PredefinedModule):
    def __init__(self, modules_list, sam2_predictor, device, trace_path=None):
        super().__init__("get_object_3d_position", trace_path)
        self.modules_list = modules_list
        self.sam2_predictor = sam2_predictor
        self.device = device

    def get_position(self, image, extrinsics, intrinsics, depth_map, image_id, object_description):
        """
        Gets the 3D position of an object in a specific image using segmentation mask.
        
        Args:
            image (image): The image containing the object
            extrinsics (np.array): Camera extrinsic matrices for all images. shape: (N, 4, 4)
            intrinsics (np.array): Camera intrinsic matrices for all images. shape: (N, 3, 3)
            depth_map (np.array): Depth maps for all images. shape: (N, H, W)
            image_id (int): Index of the image containing the object
            object_description (string): Description of the object to locate
            
        Returns:
            np.array: 3D coordinates [x, y, z] of the object in world coordinates
        """
        self.write_trace(f"<p>Getting 3D position of '{object_description}' in image {image_id} using segmentation</p>")
        
        # 1. Locate the object in the specified image
        bbox = self.modules_list.modules_dict["loc"].locate_bboxs(image, object_description)
        
        if not bbox or len(bbox) != 4:
            self.write_trace(f"<p>Error: Could not locate '{object_description}' in image {image_id}</p>")
            return None
        
        self.write_trace(f"<p>Located '{object_description}' at bbox: {bbox}</p>")
        
        # 2. Use SAM2 to segment the object mask
        object_mask = self._segment_object_mask(image, bbox, object_description)
        
        if object_mask is None:
            self.write_trace(f"<p>Error: Could not segment '{object_description}' mask</p>")
            return None
        
        # 3. Extract 3D points using the mask
        object_3d_position, _ = self._extract_3d_position_from_mask(
            object_mask, extrinsics, intrinsics, depth_map, image_id, object_description
        )
        
        if object_3d_position is None:
            self.write_trace(f"<p>Error: Could not determine 3D position from mask</p>")
            return None
        
        self.write_trace(f"<p>Success: Object 3D position from segmentation: [{object_3d_position[0]:.3f}, {object_3d_position[1]:.3f}, {object_3d_position[2]:.3f}]</p>")
        
        return object_3d_position
    
    def _segment_object_mask(self, image, bbox, object_description):
        """
        Use SAM2 to segment the object mask based on the bounding box.
        
        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2] bounding box coordinates
            object_description: string description of the object
            
        Returns:
            numpy array: Binary mask of the object, or None if segmentation fails
        """
        # Convert PIL image to numpy array
        image_np = np.array(image)
        H, W, _ = image_np.shape
        
        # Set image for SAM2
        self.sam2_predictor.set_image(image_np)
        
        # Convert bbox to format expected by SAM2
        x1, y1, x2, y2 = bbox
        box_xyxy_np = np.array([x1, y1, x2, y2])
        
        self.write_trace(f"<p>Using SAM2 to segment '{object_description}' with bbox: [{x1}, {y1}, {x2}, {y2}]</p>")
        
        # Predict mask using SAM2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy_np,
            multimask_output=False,  # Use single best mask
        )
        
        # Get the best mask and ensure it's boolean type
        object_mask = masks[0]  # Shape: (H, W)
        mask_score = scores[0]
        
        # Convert mask to boolean if it's not already
        if object_mask.dtype != bool:
            object_mask = object_mask.astype(bool)
            self.write_trace(f"<p>Converted mask from {masks[0].dtype} to boolean</p>")
        
        self.write_trace(f"<p>Segmentation completed with score: {mask_score:.3f}</p>")
        self.write_trace(f"<p>Mask covers {np.sum(object_mask)} pixels out of {object_mask.size} total</p>")
        
        # Create visualization
        masked_image = self._create_masked_visualization(image_np, object_mask, bbox, object_description, mask_score)
        if masked_image is not None:
            from PIL import Image as PILImage
            masked_html = html_embed_image(PILImage.fromarray(masked_image))
            self.write_trace(f"<p>Segmented {object_description}:</p>")
            self.write_trace(masked_html)
        
        return object_mask

    
    def _extract_3d_position_from_mask(self, object_mask, extrinsics, intrinsics, depth_map, image_id, object_description):
        """
        Extract 3D position using the segmentation mask.
        
        Args:
            object_mask: Binary mask of the object
            extrinsics, intrinsics, depth_map: Geometry information
            image_id: Index of the current image
            object_description: Description for logging
            
        Returns:
            np.array: 3D coordinates [x, y, z] or None if extraction fails
        """
    
        # Get current image's depth map
        current_depth_map = np.array(depth_map[image_id])
        if len(current_depth_map.shape) == 3 and current_depth_map.shape[-1] == 1:
            current_depth_map = current_depth_map.squeeze(-1)
        
        depth_height, depth_width = current_depth_map.shape
        mask_height, mask_width = object_mask.shape
        
        # Ensure mask is boolean type
        if object_mask.dtype != bool:
            object_mask = object_mask.astype(bool)
        
        # Resize mask to match depth map if necessary
        if (mask_height, mask_width) != (depth_height, depth_width):
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray(object_mask.astype(np.uint8) * 255)
            mask_img_resized = mask_img.resize((depth_width, depth_height), resample=PILImage.Resampling.NEAREST)
            object_mask = np.array(mask_img_resized) > 0

        # Get world coordinates for the entire depth map
        from vggt.utils.geometry import depth_to_world_coords_points
        extrinsic = np.array(extrinsics[image_id])
        intrinsic = np.array(intrinsics[image_id])
        
        world_coords_map, _, _ = depth_to_world_coords_points(
            current_depth_map, extrinsic, intrinsic
        )
        

        mask_points_3d = world_coords_map[object_mask]
        valid_mask = np.isfinite(mask_points_3d).all(axis=1)
        valid_points = mask_points_3d[valid_mask]
        
        if len(valid_points) == 0:
            self.write_trace(f"<p>No valid 3D points found in mask for '{object_description}'</p>")
            return None, None
        
        # Calculate robust position using median to handle outliers
        if len(valid_points) >= 3:
            object_3d_position = np.median(valid_points, axis=0)
            self.write_trace(f"<p>Used median of {len(valid_points)} points</p>")
        else:
            object_3d_position = np.mean(valid_points, axis=0)
            self.write_trace(f"<p>Used mean of {len(valid_points)} points</p>")
        
        return object_3d_position, valid_points
            
    
    def get_point_cloud(self, image, extrinsics, intrinsics, depth_map, image_id, object_description):
        """
        Gets the 3D point cloud of an object in a specific image using segmentation mask.
        """
        self.write_trace(f"<p>Getting 3D point cloud of '{object_description}' in image {image_id} using segmentation</p>")
        
        # 1. Locate the object
        bbox = self.modules_list.modules_dict["loc"].locate_bboxs(image, object_description)
        if not bbox or len(bbox) != 4:
            self.write_trace(f"<p>Error: Could not locate '{object_description}' in image {image_id}</p>")
            return None

        # 2. Segment the object mask
        object_mask = self._segment_object_mask(image, bbox, object_description)
        if object_mask is None:
            self.write_trace(f"<p>Error: Could not segment '{object_description}' mask</p>")
            return None

        # 3. Extract 3D points using the mask
        _, point_cloud = self._extract_3d_position_from_mask(
            object_mask, extrinsics, intrinsics, depth_map, image_id, object_description
        )

        if point_cloud is None:
            self.write_trace(f"<p>Error: Could not determine 3D point cloud from mask</p>")
            return None
        
        return point_cloud

    def _create_masked_visualization(self, image, mask, bbox, object_description, score):
        """Create visualization showing the segmented object."""
        from PIL import Image as PILImage, ImageDraw, ImageFont
        
        # Ensure mask is boolean type
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Create overlay image
        pil_image = PILImage.fromarray(image)
        overlay = PILImage.new('RGBA', pil_image.size, (0, 0, 0, 0))
        
        # Convert mask to RGBA
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask] = [255, 0, 0, 255]  # Red mask with transparency
        mask_image_pil = PILImage.fromarray(mask_rgba, mode='RGBA')
        
        overlay.paste(mask_image_pil, (0, 0), mask_image_pil)
        
        # Draw bounding box
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = [int(x) for x in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)
        
        # Add text label
        font = ImageFont.load_default()
        text = f"{object_description} ({score:.2f})"
        draw.text((x1, y1-20), text, fill=(0, 255, 0, 255), font=font)
        
        # Composite images
        pil_image = pil_image.convert('RGBA')
        composite = PILImage.alpha_composite(pil_image, overlay)
        
        return np.array(composite.convert('RGB'))

class SituatedOrientationGroundingModule(PredefinedModule):
    def __init__(self, modules_list, vqa_model, sog_model, trace_path=None):
        super().__init__("situated_orientation_grounding", trace_path)
        self.modules_list = modules_list
        self.identify_generator = Generator(vqa_model, temperature=1.0)
        self.sog_generator = Generator(sog_model, temperature=1.0)
        self.visualization_utils = VisualizationUtils(device=getattr(modules_list, 'device', 'cuda'))

    def ground_situated_orientation(self, images, extrinsics, intrinsics, depth_maps, world_points, ground_normal, ground_centroid, orientation_description):
        """Ground situated orientation description to 3D direction vector and position."""
        print(f"Situated Orientation Grounding: {orientation_description}")
        self.write_trace(f"<h2>Situated Orientation Grounding: {orientation_description}</h2>")

        # Use unified approach to identify object and select best image
        key_object, image_id, reasoning = self._identify_object_and_best_image(images, orientation_description)
        
        if not key_object or image_id is None:
            self.write_trace("<p>Error: Could not identify key object or select best image for orientation</p>")
            return None

        self.write_trace(f"<p>Identified key object: <b>{key_object}</b></p>")
        self.write_trace(f"<p>Selected image {image_id} as best view</p>")
        self.write_trace(f"<p>Selection reasoning: {reasoning}</p>")

        object_point_cloud = self.modules_list.modules_dict["get_object_3d_position"].get_point_cloud(
            images[image_id], extrinsics, intrinsics, depth_maps, image_id, key_object
        )

        if object_point_cloud is None:
            self.write_trace(f"<p>Found {key_object} in image {image_id} but failed to get 3D point cloud</p>")
            return None

        ground_normal_np = np.array(ground_normal)
        ground_centroid_np = np.array(ground_centroid)

        ground_normal_np = ground_normal_np / np.linalg.norm(ground_normal_np)

        distances = np.abs(np.dot(object_point_cloud - ground_centroid_np, ground_normal_np))
        sorted_indices = np.argsort(distances)
        percentile_30_idx = int(len(sorted_indices) * 0.3)
        selected_point_idx = sorted_indices[percentile_30_idx]
        object_3d_position = object_point_cloud[selected_point_idx]

        self.write_trace(f"<p>Object point cloud size: {len(object_point_cloud)}. Point at 30% distance from ground plane selected as origin.</p>")
        self.write_trace(f"<p>Selected origin position (30% distance): {object_3d_position}</p>")
        source_image = images[image_id]

        self.write_trace(f"<p>Object located at position: {object_3d_position} in image {image_id}</p>")
        
        world_colors = self.visualization_utils.extract_colors_from_images(world_points, images)

        filtered_world_points, filtered_colors = self.visualization_utils.filter_point_cloud_by_height(
            world_points, ground_normal, ground_centroid, height_percentile_cutoff=0.8, colors=world_colors
        )
        original_overlay_image, elevated_overlay_image, arrow_directions = self._create_directional_overlay(
            source_image, object_3d_position, extrinsics[image_id],
            intrinsics[image_id], ground_normal, filtered_world_points, filtered_colors,
        )
        
        self._current_ground_normal = np.array(ground_normal)
        selected_arrow_id = self._select_closest_arrow(original_overlay_image, elevated_overlay_image, orientation_description, key_object)
        axis_directions_3d_world = self._create_ground_plane_directions(ground_normal, extrinsics[image_id])
        fine_directions_world = self._create_fine_grained_directions_around_selected(selected_arrow_id, axis_directions_3d_world, ground_normal, extrinsics[image_id])
        original_fine_overlay_image, elevated_fine_overlay_image = self._create_fine_grained_overlay(
            source_image, object_3d_position, extrinsics[image_id], intrinsics[image_id],
            fine_directions_world, selected_arrow_id, filtered_world_points, filtered_colors
        )
        final_arrow_id = self._select_final_direction_arrow(original_fine_overlay_image, elevated_fine_overlay_image, orientation_description, key_object)

        if final_arrow_id >= len(fine_directions_world):
            self.write_trace(f"<p>Error: Invalid fine-grained arrow selection: {final_arrow_id}</p>")
            return None
        selected_direction_vector = fine_directions_world[str(final_arrow_id)]
        self.write_trace(f"<p>Selected fine-grained arrow {final_arrow_id} from base arrow {selected_arrow_id}</p>")
        self.write_trace(f"<p>Final direction vector: {selected_direction_vector}</p>")

        result = {
            "position": object_3d_position,
            "direction_vector": selected_direction_vector,
            "key_object": key_object,
            "image_id": image_id,
            "arrow_id": final_arrow_id,
            "base_arrow_id": selected_arrow_id,
        }

        self.write_trace(f"<p><b>Situated Orientation Grounding Result:</b></p>")
        self.write_trace(f"<p>Position: {object_3d_position}</p>")
        self.write_trace(f"<p>Direction Vector: {selected_direction_vector}</p>")
        self.write_trace(f"<p>Key Object: {key_object}</p>")

        return result

    def _identify_key_object(self, orientation_description):
        """Use VQA to identify the key object related to the orientation description."""
        self.write_trace("<h3>Step 1: Identifying Key Object</h3>")

        prompt = IDENTIFY_KEY_OBJECT_PROMPT.format(orientation_description=orientation_description)

        output, _ = self.identify_generator.generate(prompt)

        try:
            key_object = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].strip().lower()
            self.write_trace(f"<p>VQA response: {output}</p>")
            return key_object
        except (IndexError, AttributeError):
            self.write_trace(f"<p>Failed to parse VQA response: {output}</p>")
            return None

    def _identify_object_and_best_image(self, images, orientation_description):
        """Use VLM to identify the key object and best image in a unified approach."""
        self.write_trace("<h3>Step 1: Unified Object and Image Selection</h3>")
        
        image_ids_str = ", ".join(map(str, range(len(images))))
        prompt = UNIFIED_OBJECT_SELECTION_PROMPT.format(orientation_description=orientation_description, image_ids=image_ids_str)
        
        content = [{"type": "text", "text": prompt}]
        
        self.write_trace(f"<h4>Analyzing {len(images)} images for orientation: {orientation_description}</h4>")
        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            })

        messages = [{"role": "user", "content": content}]
        
        output, _ = self.identify_generator.generate("", messages)
        
        self.write_trace(f"<p><b>Unified Selection Prompt:</b> {orientation_description}</p>")
        self.write_trace(f"<p><b>VLM Response:</b> {output}</p>")
        
        try:
            answer_json_str = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].strip()
            answer_dict = json.loads(answer_json_str)
            
            target_object = answer_dict.get("target_object", "").lower()
            best_image_id = int(answer_dict.get("best_image_id", 0))
            reasoning = answer_dict.get("reasoning", "")
            
            # Validate the selection
            if best_image_id < 0 or best_image_id >= len(images):
                self.write_trace(f"<p>Warning: Invalid image ID {best_image_id}, using image 0</p>")
                best_image_id = 0
                
            if not target_object:
                self.write_trace(f"<p>Warning: No target object identified</p>")
                return None, None, None
            
            self.write_trace(f"<p><b>Selected Object:</b> {target_object}</p>")
            self.write_trace(f"<p><b>Best Image ID:</b> {best_image_id}</p>")
            self.write_trace(f"<p><b>Reasoning:</b> {reasoning}</p>")
            
            return target_object, best_image_id, reasoning
            
        except (IndexError, json.JSONDecodeError, ValueError) as e:
            self.write_trace(f"<p><b>Error parsing unified selection response: {e}. Output was: {output}</b></p>")
            self.write_trace(f"<p>Falling back to separate identification steps...</p>")
            key_object = self._identify_key_object(orientation_description)
            if key_object:
                object_locations = self.modules_list.modules_dict["find_obj"].find(images, [key_object])
                if key_object in object_locations:
                    return key_object, object_locations[key_object], "Fallback method used"
            return None, None, None
                

    def _create_ground_plane_directions(self, ground_normal, extrinsic):
        """Create four directions in ground plane, perpendicular to ground normal."""
        directions = {}

        ground_normal = np.array(ground_normal, dtype=np.float64)
        ground_normal = ground_normal / np.linalg.norm(ground_normal)

        extrinsic_matrix = np.array(extrinsic, dtype=np.float64)
        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])

        rotation_matrix = extrinsic_matrix[:3, :3]
        camera_to_world_rotation = rotation_matrix.T

        camera_x_world = camera_to_world_rotation[:, 0]
        camera_x_projected = camera_x_world - np.dot(camera_x_world, ground_normal) * ground_normal

        if np.linalg.norm(camera_x_projected) < 1e-6:
            self.write_trace("<p>Warning: Camera x-axis is parallel to ground normal, using fallback</p>")
            if abs(ground_normal[0]) < 0.9:
                v1 = np.cross(ground_normal, [1, 0, 0])
            else:
                v1 = np.cross(ground_normal, [0, 1, 0])
            v1 = v1 / np.linalg.norm(v1)
        else:
            v1 = camera_x_projected / np.linalg.norm(camera_x_projected)

        v2 = np.cross(ground_normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        self.write_trace(f"<p>Camera x-axis in world: [{camera_x_world[0]:.3f}, {camera_x_world[1]:.3f}, {camera_x_world[2]:.3f}]</p>")
        self.write_trace(f"<p>Projected camera x (v1): [{v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}]</p>")
        self.write_trace(f"<p>Perpendicular vector (v2): [{v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}]</p>")

        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        labels = ['0', '1', '2', '3']

        for i, (angle, label) in enumerate(zip(angles, labels)):
            direction = np.cos(angle) * v1 + np.sin(angle) * v2
            directions[label] = direction

        return directions

    def _create_directional_overlay(self, image, object_position, extrinsic, intrinsic, ground_normal, filtered_world_points, filtered_colors):
        """Create an overlay image with directional arrows around the object position (dual view)."""
        self.write_trace("<h3>Step 3: Creating Directional Overlay (Dual View)</h3>")

        rgb_image = np.array(image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        world_origin = np.array(object_position, dtype=np.float64)
        extrinsic_matrix = np.array(extrinsic, dtype=np.float64)
        intrinsic_np = np.array(intrinsic, dtype=np.float64)

        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])

        origin_3d_camera = self.visualization_utils.world_to_camera_coordinates(world_origin, extrinsic_matrix)

        self.write_trace(f"<p>Object world position: [{world_origin[0]:.3f}, {world_origin[1]:.3f}, {world_origin[2]:.3f}]</p>")
        self.write_trace(f"<p>Object camera position: [{origin_3d_camera[0]:.3f}, {origin_3d_camera[1]:.3f}, {origin_3d_camera[2]:.3f}]</p>")

        if origin_3d_camera[2] <= 0:
            self.write_trace(f"<p>Error: Object is behind camera (depth: {origin_3d_camera[2]:.3f})</p>")
            return None, None, {}

        axis_directions_3d_world = self._create_ground_plane_directions(ground_normal, extrinsic_matrix)

        axis_directions_3d_camera = {}
        for axis_name, world_direction in axis_directions_3d_world.items():
            camera_direction = self.visualization_utils.world_direction_to_camera(world_direction, extrinsic_matrix)
            axis_directions_3d_camera[axis_name] = camera_direction
            self.write_trace(f"<p>Direction {axis_name} - World: [{world_direction[0]:.3f}, {world_direction[1]:.3f}, {world_direction[2]:.3f}] -> Camera: [{camera_direction[0]:.3f}, {camera_direction[1]:.3f}, {camera_direction[2]:.3f}]</p>")

        pixel_arrow_length = 80
        min_pixel_length = 50

        depth = origin_3d_camera[2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        focal_length = (fx + fy) / 2.0

        target_3d_length = (pixel_arrow_length * depth) / focal_length
        min_3d_length = (min_pixel_length * depth) / focal_length

        arrow_length_3d = max(target_3d_length, min_3d_length, 0.4)

        self.write_trace(f"<p>Arrow 3D length: {arrow_length_3d:.3f} units (camera coordinates)</p>")

        all_points_2d = []
        origin_pixel = self.visualization_utils.camera_3d_to_pixel(origin_3d_camera, intrinsic_np)
        if origin_pixel is not None:
            all_points_2d.append(origin_pixel)

        arrow_tip_ratio = 0.15
        font_scale = 0.7
        thickness = 2
        background_padding = 2

        for axis_name in ['0', '1', '2', '3']:
            direction_3d_camera = axis_directions_3d_camera[axis_name]
            endpoint_3d_camera = origin_3d_camera + direction_3d_camera * arrow_length_3d
            endpoint_pixel = self.visualization_utils.camera_3d_to_pixel(endpoint_3d_camera, intrinsic_np)

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

                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = cv2.getTextSize(axis_name, font, font_scale, thickness)
                text_x = int(label_pixel[0] - text_width // 2)
                text_y = int(label_pixel[1] + text_height // 2)

                label_top_left = np.array([text_x - background_padding, text_y - text_height - background_padding])
                label_bottom_right = np.array([text_x + text_width + background_padding, text_y + baseline + background_padding])
                all_points_2d.extend([label_top_left, label_bottom_right])

        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
        if all_points_2d:
            points_arr = np.array(all_points_2d)
            min_x, min_y = np.min(points_arr, axis=0)
            max_x, max_y = np.max(points_arr, axis=0)
            h, w, _ = bgr_image.shape

            pad_left = int(max(0, -min_x))
            pad_top = int(max(0, -min_y))
            pad_right = int(max(0, max_x - w))
            pad_bottom = int(max(0, max_y - h))

            if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
                new_w = w + pad_left + pad_right
                new_h = h + pad_top + pad_bottom
                padded_image = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
                padded_image[pad_top:pad_top+h, pad_left:pad_left+w] = bgr_image
                bgr_image = padded_image
                self.write_trace(f"<p>Padded image to {new_w}x{new_h} with padding (T,B,L,R): ({pad_top}, {pad_bottom}, {pad_left}, {pad_right})</p>")

        offset_2d = (pad_left, pad_top)

        colors_bgr = {
            '0': (0, 0, 255),      # Red
            '1': (255, 0, 0),      # Blue  
            '2': (0, 180, 200),    # Yellow (reduced saturation)
            '3': (0, 180, 0),      # Green (reduced saturation)
        }

        self._current_colors_bgr = colors_bgr

        arrow_directions_world = []
        for axis_name in ['0', '1', '2', '3']:
            direction_3d_camera = axis_directions_3d_camera[axis_name]
            direction_3d_world = axis_directions_3d_world[axis_name]
            color = colors_bgr[axis_name]

            success = self.visualization_utils.draw_arrow_3d_on_image(bgr_image, origin_3d_camera, direction_3d_camera, arrow_length_3d, color, intrinsic_np, offset_2d=offset_2d)

            self.visualization_utils.add_text_label_3d_on_image(bgr_image, origin_3d_camera, direction_3d_camera, arrow_length_3d, axis_name, color, intrinsic_np, offset_2d=offset_2d, font_scale=font_scale, thickness=thickness, background_padding=background_padding)
            self.write_trace(f"<p>Drew arrow {axis_name}</p>")

            arrow_directions_world.append(direction_3d_world.tolist())

        origin_pixel = self.visualization_utils.camera_3d_to_pixel(origin_3d_camera, intrinsic_np)
        if origin_pixel is not None:
            origin_pixel_int = (int(origin_pixel[0] + offset_2d[0]), int(origin_pixel[1] + offset_2d[1]))
            cv2.circle(bgr_image, origin_pixel_int, 5, (0, 0, 0), -1)
            self.write_trace(f"<p>Origin projected to pixel: [{origin_pixel[0]:.1f}, {origin_pixel[1]:.1f}]</p>")

        overlay_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        overlay_image = Image.fromarray(overlay_rgb)

        self.write_trace(f"<p>Successfully created overlay with {len(arrow_directions_world)} directional arrows</p>")

        overlay_html = html_embed_image(overlay_image, size=500)
        self.write_trace("<p>Original overlay image with directional arrows:</p>")
        self.write_trace(overlay_html)

        self.write_trace("<h4>Creating Elevated Camera View</h4>")

        elevated_extrinsic = self.visualization_utils.calculate_elevated_camera_pose(
            extrinsic_matrix, np.array(object_position), np.array(ground_normal), 45.0
        )

        image_size = (image.width, image.height)
        arrows_info = {
            'origin': object_position,
            'directions': axis_directions_3d_world,
            'arrow_length': arrow_length_3d,
            'font_scale': 0.7
        }
        rendered_image = self.visualization_utils.render_point_cloud_to_image(
            filtered_world_points, elevated_extrinsic, intrinsic_np, image_size,
            background_color=(255, 255, 255), point_size=2,
            pre_extracted_colors=filtered_colors,
            arrows_info=arrows_info
        )

        offset_2d = (0, 0)
        if arrows_info is not None:
            _, (pad_left, pad_top, pad_right, pad_bottom) = self.visualization_utils._calculate_required_size_with_arrows(
                image_size, elevated_extrinsic, intrinsic_np, arrows_info
            )
            offset_2d = (pad_left, pad_top)

        elevated_overlay = self.visualization_utils.overlay_arrows_on_rendered_image(
            rendered_image, object_position, axis_directions_3d_world, elevated_extrinsic, intrinsic_np,
            arrow_length_3d=arrow_length_3d, colors=self._current_colors_bgr, font_scale=0.7, offset_2d=offset_2d
        )

        elevated_overlay_pil = Image.fromarray(cv2.cvtColor(elevated_overlay, cv2.COLOR_BGR2RGB))

        elevated_html = html_embed_image(elevated_overlay_pil, size=500)
        self.write_trace("<p>Elevated view overlay image with directional arrows:</p>")
        self.write_trace(elevated_html)

        original_array = np.array(overlay_image)
        elevated_overlay_rgb = cv2.cvtColor(elevated_overlay, cv2.COLOR_BGR2RGB)
        elevated_overlay_image = Image.fromarray(elevated_overlay_rgb)
        
        # Create concatenated view for trace display only
        concatenated_array = self.visualization_utils.concatenate_images_horizontally(original_array, elevated_overlay_rgb)
        concatenated_image = Image.fromarray(concatenated_array)

        concatenated_html = html_embed_image(concatenated_image, size=800)
        self.write_trace("<p>Concatenated view (Original | Elevated):</p>")
        self.write_trace(concatenated_html)

        return overlay_image, elevated_overlay_image, arrow_directions_world

    def _select_closest_arrow(self, original_overlay_image, elevated_overlay_image, orientation_description, key_object):
        """Use VLM to select the single closest arrow from the initial 4 directions using two separate views"""
        self.write_trace(f"<h3>Step 4a: Selecting Closest Single Arrow</h3>")

        # Convert original view to base64
        buffered_original = io.BytesIO()
        original_overlay_image.save(buffered_original, format="PNG")
        base64_original = base64.b64encode(buffered_original.getvalue()).decode("utf-8")
        
        # Convert elevated view to base64
        buffered_elevated = io.BytesIO()
        elevated_overlay_image.save(buffered_elevated, format="PNG")
        base64_elevated = base64.b64encode(buffered_elevated.getvalue()).decode("utf-8")

        prompt = COARSE_PROMPT.format(orientation_description=orientation_description)
        self.write_trace(f"<p>Target direction: {orientation_description}</p>")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_original}"}
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_elevated}"}
                    }
                ]
            }
        ]

        output, _ = self.sog_generator.generate("", messages)
        try:
            answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].strip()
            arrow_id = int(answer)

            if arrow_id not in [0, 1, 2, 3]:
                raise ValueError("Arrow ID must be 0, 1, 2, or 3")

            self.write_trace(f"<p>VLM response: {output}</p>")
            self.write_trace(f"<p>Selected closest arrow: {arrow_id}</p>")
            return arrow_id

        except (IndexError, ValueError, AttributeError) as e:
            self.write_trace(f"<p>Failed to parse VLM response: {output}. Error: {e}</p>")
            self.write_trace(f"<p>Using fallback arrow ID: 0</p>")
            return 0

    def _create_fine_grained_directions_around_selected(self, selected_arrow_id, axis_directions_3d_world, ground_normal, extrinsic):
        """Create 5 fine-grained directions around the selected arrow: base + ±22.5° + ±45°"""

        base_direction = axis_directions_3d_world[str(selected_arrow_id)]

        base_angle_deg = selected_arrow_id * 90.0

        angle_offsets = [-45, -22.5, 0, 22.5, 45]
        fine_angles_deg = [base_angle_deg + offset for offset in angle_offsets]

        angles_rad = np.radians(fine_angles_deg)

        ground_normal_norm = np.array(ground_normal, dtype=np.float64)
        ground_normal_norm = ground_normal_norm / np.linalg.norm(ground_normal_norm)

        extrinsic_matrix = np.array(extrinsic, dtype=np.float64)
        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])

        rotation_matrix = extrinsic_matrix[:3, :3]
        camera_to_world_rotation = rotation_matrix.T
        
        # Camera coordinate system x-axis in world coordinates
        camera_x_world = camera_to_world_rotation[:, 0]  # [1, 0, 0] in camera -> world
        
        # Project camera x-axis onto ground plane (make it perpendicular to ground normal)
        camera_x_projected = camera_x_world - np.dot(camera_x_world, ground_normal_norm) * ground_normal_norm

        if np.linalg.norm(camera_x_projected) < 1e-6:
            if abs(ground_normal_norm[0]) < 0.9:
                v1 = np.cross(ground_normal_norm, [1, 0, 0])
            else:
                v1 = np.cross(ground_normal_norm, [0, 1, 0])
            v1 = v1 / np.linalg.norm(v1)
        else:
            v1 = camera_x_projected / np.linalg.norm(camera_x_projected)

        v2 = np.cross(ground_normal_norm, v1)
        v2 = v2 / np.linalg.norm(v2)

        fine_directions = {}
        for i, angle in enumerate(angles_rad):
            direction = np.cos(angle) * v1 + np.sin(angle) * v2
            fine_directions[str(i)] = direction

        self.write_trace(f"<p>Created 5 fine-grained directions around arrow {selected_arrow_id} (22.5° spacing):</p>")
        for i, direction in fine_directions.items():
            angle_deg = fine_angles_deg[int(i)]
            offset_deg = angle_offsets[int(i)]
            self.write_trace(f"<p>  Fine direction {i} (base{offset_deg:+.1f}° = {angle_deg:.1f}°): [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]</p>")

        return fine_directions

    def _create_fine_grained_overlay(self, image, object_position, extrinsic, intrinsic,
                                   fine_directions_world, selected_arrow_id, filtered_world_points, filtered_colors):
        """Create overlay with fine-grained arrows (dual view)"""
        self.write_trace(f"<h3>Step 4b: Creating Fine-Grained Overlay (Dual View)</h3>")

        rgb_image = np.array(image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        world_origin = np.array(object_position, dtype=np.float64)
        extrinsic_matrix = np.array(extrinsic, dtype=np.float64)
        intrinsic_np = np.array(intrinsic, dtype=np.float64)

        if extrinsic_matrix.shape == (3, 4):
            extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])

        origin_3d_camera = self.visualization_utils.world_to_camera_coordinates(world_origin, extrinsic_matrix)

        if origin_3d_camera[2] <= 0:
            self.write_trace(f"<p>Error: Fine-grained overlay origin is behind camera</p>")
            return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

        fine_directions_camera = {}
        for axis_name, world_direction in fine_directions_world.items():
            camera_direction = self.visualization_utils.world_direction_to_camera(world_direction, extrinsic_matrix)
            fine_directions_camera[axis_name] = camera_direction

        depth = origin_3d_camera[2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        focal_length = (fx + fy) / 2.0
        pixel_arrow_length = 60  # Slightly smaller for fine-grained view
        min_pixel_length = 40
        target_3d_length = (pixel_arrow_length * depth) / focal_length
        min_3d_length = (min_pixel_length * depth) / focal_length
        arrow_length_3d = max(target_3d_length, min_3d_length, 0.25)

        all_points_2d = []
        origin_pixel = self.visualization_utils.camera_3d_to_pixel(origin_3d_camera, intrinsic_np)
        if origin_pixel is not None:
            all_points_2d.append(origin_pixel)

        arrow_tip_ratio = 0.15
        font_scale = 0.5
        thickness = 1
        background_padding = 1

        for axis_name in ['0', '1', '2', '3', '4']:
            direction_3d_camera = fine_directions_camera[axis_name]
            endpoint_3d_camera = origin_3d_camera + direction_3d_camera * arrow_length_3d
            endpoint_pixel = self.visualization_utils.camera_3d_to_pixel(endpoint_3d_camera, intrinsic_np)

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

                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = cv2.getTextSize(axis_name, font, font_scale, thickness)
                text_x = int(label_pixel[0] - text_width // 2)
                text_y = int(label_pixel[1] + text_height // 2)

                label_top_left = np.array([text_x - background_padding, text_y - text_height - background_padding])
                label_bottom_right = np.array([text_x + text_width + background_padding, text_y + baseline + background_padding])
                all_points_2d.extend([label_top_left, label_bottom_right])

        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
        if all_points_2d:
            points_arr = np.array(all_points_2d)
            min_x, min_y = np.min(points_arr, axis=0)
            max_x, max_y = np.max(points_arr, axis=0)
            h, w, _ = bgr_image.shape

            pad_left = int(max(0, -min_x))
            pad_top = int(max(0, -min_y))
            pad_right = int(max(0, max_x - w))
            pad_bottom = int(max(0, max_y - h))

            if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
                new_w = w + pad_left + pad_right
                new_h = h + pad_top + pad_bottom
                padded_image = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
                padded_image[pad_top:pad_top+h, pad_left:pad_left+w] = bgr_image
                bgr_image = padded_image
                self.write_trace(f"<p>Padded fine-grained image with (T,B,L,R): ({pad_top}, {pad_bottom}, {pad_left}, {pad_right})</p>")

        offset_2d = (pad_left, pad_top)

        fine_colors_bgr = {
            '0': (0, 100, 255),    # Orange-red
            '1': (100, 0, 255),    # Purple  
            '2': (255, 100, 0),    # Cyan (center/base direction)
            '3': (0, 150, 100),    # Dark green
            '4': (150, 150, 0),    # Teal
        }

        self._current_fine_colors_bgr = fine_colors_bgr
        for axis_name in ['0', '1', '2', '3', '4']:
            direction_3d_camera = fine_directions_camera[axis_name]
            color = fine_colors_bgr[axis_name]

            success = self.visualization_utils.draw_arrow_3d_on_image(bgr_image, origin_3d_camera, direction_3d_camera,
                                        arrow_length_3d, color, intrinsic_np, line_thickness=2, offset_2d=offset_2d)

            if success:
                self.visualization_utils.add_text_label_3d_on_image(bgr_image, origin_3d_camera, direction_3d_camera,
                                      arrow_length_3d, axis_name, color, intrinsic_np, offset_2d=offset_2d,
                                      font_scale=font_scale, thickness=thickness, background_padding=background_padding)

        origin_pixel = self.visualization_utils.camera_3d_to_pixel(origin_3d_camera, intrinsic_np)
        if origin_pixel is not None:
            origin_pixel_int = (int(origin_pixel[0] + offset_2d[0]), int(origin_pixel[1] + offset_2d[1]))
            cv2.circle(bgr_image, origin_pixel_int, 4, (0, 0, 0), -1)

        overlay_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        fine_overlay_image = Image.fromarray(overlay_rgb)

        self.write_trace(f"<p>Created fine-grained overlay around selected arrow {selected_arrow_id}</p>")

        overlay_html = html_embed_image(fine_overlay_image, size=500)
        self.write_trace("<p>Original fine-grained overlay image:</p>")
        self.write_trace(overlay_html)

        self.write_trace("<h4>Creating Elevated View for Fine-Grained Overlay</h4>")

        ground_normal = self._current_ground_normal

        elevated_extrinsic = self.visualization_utils.calculate_elevated_camera_pose(
            extrinsic_matrix, np.array(object_position), np.array(ground_normal), 45.0
        )

        image_size = (image.width, image.height)
        arrows_info = {
            'origin': object_position,
            'directions': fine_directions_world,
            'arrow_length': arrow_length_3d,
            'font_scale': 0.5
        }
        rendered_image = self.visualization_utils.render_point_cloud_to_image(
            filtered_world_points, elevated_extrinsic, intrinsic_np, image_size,
            background_color=(255, 255, 255), point_size=2,
            pre_extracted_colors=filtered_colors,
            arrows_info=arrows_info
        )

        offset_2d = (0, 0)
        if arrows_info is not None:
            _, (pad_left, pad_top, pad_right, pad_bottom) = self.visualization_utils._calculate_required_size_with_arrows(
                image_size, elevated_extrinsic, intrinsic_np, arrows_info
            )
            offset_2d = (pad_left, pad_top)

        elevated_fine_overlay = self.visualization_utils.overlay_arrows_on_rendered_image(
            rendered_image, object_position, fine_directions_world, elevated_extrinsic, intrinsic_np,
            arrow_length_3d=arrow_length_3d, colors=self._current_fine_colors_bgr, font_scale=0.5, offset_2d=offset_2d
        )

        elevated_fine_overlay_pil = Image.fromarray(cv2.cvtColor(elevated_fine_overlay, cv2.COLOR_BGR2RGB))

        elevated_html = html_embed_image(elevated_fine_overlay_pil, size=500)

        self.write_trace("<p>Elevated fine-grained overlay image:</p>")
        self.write_trace(elevated_html)

        original_fine_array = np.array(fine_overlay_image)
        elevated_fine_overlay_rgb = cv2.cvtColor(elevated_fine_overlay, cv2.COLOR_BGR2RGB)
        elevated_fine_overlay_image = Image.fromarray(elevated_fine_overlay_rgb)
        
        # Create concatenated view for trace display only
        concatenated_fine_array = self.visualization_utils.concatenate_images_horizontally(original_fine_array, elevated_fine_overlay_rgb)
        concatenated_fine_image = Image.fromarray(concatenated_fine_array)

        concatenated_fine_html = html_embed_image(concatenated_fine_image, size=800)
        self.write_trace("<p>Concatenated fine-grained view (Original | Elevated):</p>")
        self.write_trace(concatenated_fine_html)

        return fine_overlay_image, elevated_fine_overlay_image

    def _select_final_direction_arrow(self, original_fine_overlay_image, elevated_fine_overlay_image, orientation_description, key_object):
        """Use VLM to select final arrow from fine-grained directions using two separate views"""
        self.write_trace(f"<h3>Step 4c: Selecting Final Direction</h3>")

        # Convert original fine view to base64
        buffered_original = io.BytesIO()
        original_fine_overlay_image.save(buffered_original, format="PNG")
        base64_original = base64.b64encode(buffered_original.getvalue()).decode("utf-8")
        
        # Convert elevated fine view to base64
        buffered_elevated = io.BytesIO()
        elevated_fine_overlay_image.save(buffered_elevated, format="PNG")
        base64_elevated = base64.b64encode(buffered_elevated.getvalue()).decode("utf-8")

        prompt = FINE_PROMPT.format(orientation_description=orientation_description)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_original}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_elevated}"}
                    }
                ]
            }
        ]

        output, _ = self.sog_generator.generate("", messages)
        try:
            arrow_id = int(re.findall(r"<answer>(\d+)</answer>", output, re.DOTALL)[0])
            self.write_trace(f"<p>VLM response: {output}</p>")
            self.write_trace(f"<p>Selected final arrow ID: {arrow_id}</p>")
            return arrow_id
        except (IndexError, ValueError, AttributeError):
            self.write_trace(f"<p>Failed to parse VLM response: {output}</p>")
            return 0

class GroundPlaneDetectionModule(PredefinedModule):
    def __init__(self, modules_list, groundingdino_model, sam2_predictor, vggt_model, device, trace_path=None, 
                 image_resize=(640, 480), prompt="Floor"):
        super().__init__("ground_plane_detection", trace_path)
        self.modules_list = modules_list
        self.groundingdino_model = groundingdino_model
        self.sam2_predictor = sam2_predictor
        self.vggt_model = vggt_model
        self.device = device
        self.TEXT_PROMPT = prompt
        self.BOX_THRESHOLD = 0.25
        self.TEXT_THRESHOLD = 0.25  
    
    def detect_ground_plane(self, images, image_paths=None):
        """
        Detect ground plane from multiple images.
        
        Args:
            images: List of PIL Image objects (preprocessed)
            image_paths: Optional list of image paths for display
            
        Returns:
            dict: Contains ground plane normal, centroid, and visualization data
        """
        self.write_trace("<h2>Ground Plane Detection</h2>")
        
        if not isinstance(images, list):
            images = [images]
        
        self.write_trace(f"<h3>Step 1: Finding image with '{self.TEXT_PROMPT}' using VQA</h3>")
        
        # Use find_obj to identify the best image first
        object_locations = self.modules_list.modules_dict["find_obj"].find(images, [self.TEXT_PROMPT])
        
        best_image_idx = None
        best_box = None
        best_logit = None

        prompt_lower = self.TEXT_PROMPT.lower()
        if prompt_lower in object_locations:
            image_id = object_locations[prompt_lower]
            self.write_trace(f"<p>VQA identified '{self.TEXT_PROMPT}' in image {image_id}. Running detector on this image.</p>")
            
            img = images[image_id]
            
            # Transform for GroundingDINO
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_transformed, _ = transform(img, None)
            
            # Predict with GroundingDINO
            formatted_prompt = f"{self.TEXT_PROMPT.replace(' ', '-')} ."
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=img_transformed,
                    caption=formatted_prompt,
                    box_threshold=self.BOX_THRESHOLD,
                    text_threshold=self.TEXT_THRESHOLD,
                    device=str(self.device)
                )
            
            if len(boxes) > 0:
                self.write_trace(f"<p>GroundingDINO found {len(boxes)} boxes for '{self.TEXT_PROMPT}' in image {image_id}.</p>")
                max_idx = torch.argmax(logits)
                best_image_idx = image_id
                best_box = boxes[max_idx].to(self.device)
                best_logit = logits[max_idx].to(self.device)
            else:
                self.write_trace(f"<p>GroundingDINO found no boxes for '{self.TEXT_PROMPT}' in image {image_id}.</p>")
        else:
            self.write_trace(f"<p>VQA did not find '{self.TEXT_PROMPT}' in any image.</p>")
        
        # Step 2: Use SAM2 to segment the ground mask from the best box
        ground_mask = None
        masked_image = None
        
        if best_box is not None:
            self.write_trace(f"<h3>Step 2: Best ground detection in image {best_image_idx + 1} with confidence {best_logit.item():.3f}</h3>")
            
            # Step 3: Use SAM2 to segment the ground mask
            best_image = np.array(images[best_image_idx])
            H, W, _ = best_image.shape
            
            # Set image for SAM2
            self.sam2_predictor.set_image(best_image)
            
            # Convert box format
            from groundingdino.util import box_ops
            best_box = best_box.to(self.device)
            scale_tensor = torch.tensor([W, H, W, H], dtype=best_box.dtype, device=self.device)
            box_xyxy = box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0)) * scale_tensor
            box_xyxy_np = box_xyxy.detach().cpu().numpy()[0]
            
            # Predict mask
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy_np,
                multimask_output=False,
            )
            
            ground_mask = masks[0]  # Shape: (H, W)
            masked_image = self._create_masked_image(best_image, ground_mask, box_xyxy_np, best_logit)

        else:
            self.write_trace("<p>No ground boxes found, will use entire scene for PCA</p>")
            
        if masked_image is not None:
            self.write_trace("<h3>Selected Image with Ground Mask</h3>")
            from PIL import Image as PILImage
            masked_html = html_embed_image(PILImage.fromarray(masked_image))
            self.write_trace(masked_html)
            
        self.write_trace("<h3>Step 3: Running VGGT 3D reconstruction</h3>")
        
        # Convert images to tensor for VGGT
        from torchvision.transforms import ToTensor
        to_tensor = ToTensor()
        image_tensors = []
        for img in images:
            img_tensor = to_tensor(img).to(self.device)
            image_tensors.append(img_tensor)
        image_tensors = torch.stack(image_tensors)
        
        # Run VGGT
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = self.vggt_model(image_tensors)
        
        # Extract camera parameters
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], image_tensors.shape[-2:])
        depth_maps = predictions["depth"]
        
        # Convert to numpy
        extrinsics_np = extrinsics.cpu().numpy().squeeze(0)
        intrinsics_np = intrinsics.cpu().numpy().squeeze(0)
        depth_maps_np = depth_maps.cpu().numpy().squeeze(0)
        
        # Get world points
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        world_points_all = unproject_depth_map_to_point_map(depth_maps_np, extrinsics_np, intrinsics_np)
        
        # Step 5: Extract ground points using mask
        if ground_mask is not None and best_image_idx is not None:
            depth_map_best = depth_maps_np[best_image_idx]
            
            depth_h, depth_w = depth_map_best.shape[:2]
            if ground_mask.dtype != bool:
                ground_mask = ground_mask.astype(bool)
                
            if ground_mask.shape != (depth_h, depth_w):
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(ground_mask.astype(np.uint8) * 255)
                mask_img_resized = mask_img.resize((depth_w, depth_h), resample=PILImage.Resampling.NEAREST)
                ground_mask = np.array(mask_img_resized) > 0
            
            world_points_best = world_points_all[best_image_idx]
            ground_points = world_points_best[ground_mask]
        else:
            ground_points = world_points_all.reshape(-1, 3)
        
        valid_mask = np.isfinite(ground_points).all(axis=1)
        ground_points_valid = ground_points[valid_mask]
        
        if len(ground_points_valid) < 3:
            self.write_trace("<p>Error: Not enough valid points for PCA analysis</p>")
            return None
        
        centroid = ground_points_valid.mean(axis=0)
        centered_points = ground_points_valid - centroid
        
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        ground_normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Ensure normal points upward, in default setting upward in real world is -Y.
        if ground_normal[1] > 0:
            ground_normal *= -1

        return {
            "ground_normal": ground_normal,
            "ground_centroid": centroid,
            "num_points_used": len(ground_points_valid),
            "best_image_idx": best_image_idx,
            "confidence": best_logit.item() if best_logit is not None else 0.0,
            "has_ground_mask": ground_mask is not None
        }
    
    def _create_masked_image(self, image, mask, box, confidence):
        """Create visualization of image with ground mask overlay."""
        from PIL import Image as PILImage, ImageDraw, ImageFont
        pil_image = PILImage.fromarray(image)
        overlay = PILImage.new('RGBA', pil_image.size, (0, 0, 0, 0))
        
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask] = [0, 255, 0, 255]  # Green mask with 100 alpha
        mask_image_pil = PILImage.fromarray(mask_rgba, mode='RGBA')
        
        overlay.paste(mask_image_pil, (0, 0), mask_image_pil)
        
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
        
        font = ImageFont.load_default()
        text = f"Ground ({confidence.item():.2f})"
        draw.text((x1, y1-25), text, fill=(0, 255, 0, 255), font=font)
        
        pil_image = pil_image.convert('RGBA')
        composite = PILImage.alpha_composite(pil_image, overlay)
        return np.array(composite.convert('RGB'))

class ModulesList:
    def __init__(self, models_path=None, trace_path=None, dataset="mmsi-bench", vqa_model="gemini-3.1-flash-lite-preview", sog_model="gemini-3.1-flash-lite-preview"):
        set_devices()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.vqa_call_count = 0
        self.find_obj_call_count = 0
        if dataset == "mmsi-bench":
            self.grounding_dino = load_model(
                f"{models_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                f"{models_path}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
            self.vggt = None
            print("Initializing and loading VGGT model...")
            self.vggt_model = VGGT()
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model_dir = os.path.join(_project_root, 'vggt', 'checkpoints')
            os.makedirs(model_dir, exist_ok=True)
            self.vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, model_dir=model_dir))
            self.vggt_model.eval()
            self.vggt_model = self.vggt_model.to(self.device)
            print("VGGT Initialized")
            
            # Initialize SAM2 for ground plane detection
            self.sam2_checkpoint = f"{models_path}/sam2/checkpoints/sam2.1_hiera_large.pt"
            self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam2_predictor = SAM2ImagePredictor(
                build_sam2(
                    self.sam2_model_cfg, self.sam2_checkpoint, device=self.device
                )
            )
        else:
            raise ValueError(f"Dataset '{dataset}' not supported. Supported datasets: mmsi-bench")

        self.modules = self.get_module_list(self.dataset, trace_path, vqa_model, sog_model)
        self.module_names = [module.name for module in self.modules]
        self.modules_dict = {module.name: module for module in self.modules}
        self.module_executes = self.get_module_executes(self.dataset)

    def get_module_executes(self, dataset):
        executes = {}
        if dataset == 'mmsi-bench':
            executes["loc"] = self.modules_dict["loc"].locate_bboxs
            executes["find_obj"] = self.modules_dict["find_obj"].find
            executes["relative_cam_movement"] = self.modules_dict["relative_cam_movement"].calculate
            executes["get_geo_info"] = self.modules_dict["get_geo_info"].extract
            executes["relative_object_position"] = self.modules_dict["relative_object_position"].calculate
            executes["calibrate_directions"] = self.modules_dict["calibrate_directions"].calibrate
            executes["calibrate_from_vector"] = self.modules_dict["calibrate_directions"].calibrate_from_vector
            executes["calculate_direction"] = self.modules_dict["calculate_direction"].calculate
            executes["get_object_3d_position"] = self.modules_dict["get_object_3d_position"].get_position
            executes["ground_plane_detection"] = self.modules_dict["ground_plane_detection"].detect_ground_plane
            executes["situated_orientation_grounding"] = self.modules_dict["situated_orientation_grounding"].ground_situated_orientation
        else:
            raise ValueError(f"Dataset '{dataset}' not supported. Supported datasets: mmsi-bench")

        return executes

    def get_module_list(self, dataset, trace_path, vqa_model, sog_model):
        if dataset == "mmsi-bench":
            return [
                LocateModule(dataset=dataset,grounding_dino=self.grounding_dino,trace_path=trace_path,),
                FindObjModule(trace_path=trace_path, modules_list=self, vqa_model=vqa_model),
                RelativeCamMovementModule(trace_path),
                GetGeoInfoModule(self.vggt_model, self.device, trace_path),
                RelativeObjectPositionModule(trace_path),
                CalibrateDirectionsModule(trace_path),
                CalculateDirectionModule(trace_path),
                GetObject3DPositionModule(modules_list=self, sam2_predictor=self.sam2_predictor, device=self.device, trace_path=trace_path),
                GroundPlaneDetectionModule(self, self.grounding_dino, self.sam2_predictor, self.vggt_model, self.device, trace_path, prompt="ground floor"),
                SituatedOrientationGroundingModule(modules_list=self, trace_path=trace_path, vqa_model=vqa_model, sog_model=sog_model),
            ]
        else:
            raise ValueError(f"Dataset '{dataset}' not supported. Supported datasets: mmsi-bench")

    def set_trace_path(self, trace_path):
        for module in self.modules:
            module.trace_path = trace_path


