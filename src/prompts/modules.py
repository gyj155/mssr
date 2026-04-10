MODULES_SIGNATURES_MMSI = """
\"\"\"
Returns the physical movement of camera1 relative to camera0 (i.e., how camera1 moved FROM camera0's position/orientation).

Args:
    cam_extrinsic0 (np.array): Camera extrinsic matrix for the first image. shape: (4, 4)
    cam_extrinsic1 (np.array): Camera extrinsic matrix for the second image. shape: (4, 4)
    z_is_positive_forward (bool, optional): If True, uses standard computer vision coordinate system (+X right, +Y down, +Z forward). If False, uses traditional computer graphics coordinate system (+X right, +Y up, +Z backward). Defaults to True.

Returns:
    dict: A dictionary containing the physical displacement of camera1 from camera0, expressed in camera0's local frame. For example, {"forward": 1.0, "right": 0.0, "up": 0.0, "rotate_right": 30.0, "rotate_up": 10.0, "around_x": 10.0, "around_y": 0.0, "around_z": 30.0}.
    Sign convention: positive "forward" means camera1 is physically IN FRONT of camera0 (camera moved forward in the scene). Positive "right" means camera1 is to the RIGHT of camera0. Positive "rotate_right" means camera1 is rotated clockwise (yaw right) compared to camera0.
\"\"\"
def relative_cam_movement(cam_extrinsic0, cam_extrinsic1, z_is_positive_forward=True):

\"\"\"
Returns the relative position of the an object to a camera.

Args:
    cam_extrinsic (np.array): Camera extrinsic matrix (world2cam)for the image.  shape: (4, 4)
    object_position (np.array): 3D position of the object in the world coordinate system. shape: (3,)

Returns:
    dict: A dictionary containing the relative position of the object to the camera. For example, {"forward": 1.0, "right": 0.0, "up": 0.0}. This means that the object is 1 unit forward, 0 units to the right, and 0 units up from the camera.
\"\"\"
def relative_object_position(cam_extrinsic, object_position):

\"\"\"
Returns the image ID(s) where an object appears.

Args:
    images (list): A list of images.
    object_prompts (list): A list of strings describing objects to locate. For example: ["white rug", "blue car"]
Returns:
    dict: A dictionary mapping each object prompt to the image ID where it appears most clearly. For example:
    {"white rug": 1, "blue car": 3}
Note: if an object is not found, it will not appear in the dictionary.
\"\"\"
def find_obj(images, object_prompts):

\"\"\"
Calibrates the cardinal directions (North, South, East, West) for a scene based on a known spatial relationship.
This function establishes a coordinate system for directional queries.

Args:
    world_points (np.array): The 3D point cloud of the entire scene from get_geo_info. shape: (N*H*W, 3)
    target_3d_point (list or None): The 3D coordinates [x, y, z] of the target object (e.g., "rug" in "rug is west of bed"). If None, uses the scene center (average of all points).
    center_3d_point (list or None): The 3D coordinates [x, y, z] of the reference center (e.g., "bed" in "rug is west of bed"). If None, uses the scene center (average of all points).
    known_direction (string): The known direction of target relative to center. Must be one of: "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest".
    ground_normal (np.array): Pre-computed ground plane normal vector [x, y, z].

Returns:
    dict: A calibration dictionary containing:
        - "north": np.array: Unit vector pointing north in world coordinates
        - "south": np.array: Unit vector pointing south in world coordinates
        - "east": np.array: Unit vector pointing east in world coordinates
        - "west": np.array: Unit vector pointing west in world coordinates
        - "ground_normal": np.array: Ground plane normal vector [x, y, z]
        - "ground_centroid": np.array: Ground plane centroid coordinates [x, y, z]
        
    IMPORTANT: You must preserve the COMPLETE calibration dictionary when storing for later use with calculate_direction().

Example usage: 
1. If given "The white rug is on the west side of the bed", use the rug's 3D position as target_3d_point, the bed's 3D position as center_3d_point, and "west" as known_direction.
2. If given "The lamp is northeast of the table", use: calibrate_directions(world_points, lamp_3d, table_3d, "northeast", ground_normal)

\"\"\"
def calibrate_directions(world_points, target_3d_point, center_3d_point, known_direction="west", ground_normal):

\"\"\"
Calibrates cardinal directions from a direction vector. Wrapper that calls calibrate_directions internally.

Args:
    world_points (np.array): The 3D point cloud of the entire scene. shape: (N*H*W, 3)
    direction_vector (np.array): The 3D direction vector [x, y, z] representing the known direction.
    center_3d_point (np.array or None): Reference point for the direction. If None, uses scene center.
    known_direction (string): What direction the vector represents ("north", "south", "east", "west", etc.).
    ground_normal (np.array): Pre-computed ground plane normal vector [x, y, z].

Returns:
    dict: Calibration dictionary containing cardinal directions (same as calibrate_directions).

Example: calibrate_from_vector(world_points, stairs_up_vector, stairs_position, "north", ground_normal)
\"\"\"
def calibrate_from_vector(world_points, direction_vector, center_3d_point, known_direction="north", ground_normal):

\"\"\"
Calculates the direction of a target object relative to a center object using pre-calibrated directions.

Args:
    calibration_info (dict): The COMPLETE calibration dictionary returned by calibrate_directions(). 
    target_3d_point (np.array or None): The 3D coordinates [x, y, z] of the target object to locate. If None, uses the scene center.
    center_3d_point (np.array or None): The 3D coordinates [x, y, z] of the reference center. If None, uses the scene center.
    ground_normal (np.array): Pre-computed ground plane normal vector [x, y, z]. 

Returns:
    dict: A dictionary containing:
        - "direction" (string): The direction of target relative to center. One of:
            "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", or "same location".
        - "angles_deg" (dict): The calculated angles in degrees between the direction vector and each cardinal direction:
            {"north": float, "south": float, "east": float, "west": float}

Examples: 
1. To answer "Where is the fireplace relative to the stairs?", use the fireplace's 3D position as target_3d_point and the stairs' 3D position as center_3d_point.
2. To answer "Where is the fireplace relative to the room?" or "Is the fireplace south of the room?", use the fireplace's 3D position as target_3d_point and None as center_3d_point: calculate_direction(calibration, fireplace_3d, None, ground_normal)

usage pattern:
    calibration_info = calibrate_directions(world_points, target_3d, center_3d, "southeast", ground_normal)
    direction_result = calculate_direction(calibration_info, stairs_3d, bed_3d, ground_normal)
\"\"\"
def calculate_direction(calibration_info, target_3d_point, center_3d_point, ground_normal):

\"\"\"
Gets the 3D world coordinates of an object in a specific image using pre-extracted geometric information.

Args:
    image (image): The image containing the object.
    extrinsics (np.array): Camera extrinsic matrices for all images. shape: (N, 4, 4)
    intrinsics (np.array): Camera intrinsic matrices for all images. shape: (N, 3, 3)
    depth_map (np.array): Depth maps for all images. shape: (N, H, W)
    image_id (int): Index of the image containing the object (0-based).
    object_description (string): Description of the object to locate (e.g., "white chair", "blue car").

Returns:
    np.array: The 3D coordinates [x, y, z] of the object's center in world coordinates. Returns None if object cannot be located or back-projected.

Example usage: 
    extrinsics, intrinsics, depth_maps, world_points = get_geo_info(images)
    object_3d_pos = get_object_3d_position(images[0], extrinsics, intrinsics, depth_maps, 0, "red sofa")
\"\"\"
def get_object_3d_position(image, extrinsics, intrinsics, depth_map, image_id, object_description):

\"\"\"
Grounds a situated orientation description to a specific 3D direction vector and position in the scene.
This function is designed to handle complex spatial orientation queries like "direction going up the stairs" or "direction exiting the bathtub", "the facing of a TV".

Args:
    images (list)
    extrinsics (np.array)
    intrinsics (np.array)
    depth_maps (np.array)
    world_points (np.array)
    ground_normal (np.array): Ground plane normal vector
    ground_centroid (np.array): Ground plane centroid vector
    orientation_description (string): Natural language description of the desired orientation. 
        Examples: "direction going up the stairs", "direction exiting the bathtub", "direction entering the room", "direction facing the window"

Returns:
    dict: Contains:
        - "position": np.array: 3D coordinates [x, y, z] of the reference point (usually the key object's location)
        - "direction_vector": np.array: 3D unit vector [x, y, z] representing the grounded orientation direction in world coordinates

Usage Examples:
1. For "direction exiting the bathtub":
   - Identifies "bathtub" as the key object
   - Finds bathtub location
   - Returns direction vector representing the "exiting" orientation in world coordinates

\"\"\"
def situated_orientation_grounding(images, extrinsics, intrinsics, depth_maps, world_points, ground_normal, ground_centroid, orientation_description):
"""
