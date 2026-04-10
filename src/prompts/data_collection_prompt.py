DATA_COLLECTION_PROMPT_INITIAL = """
You are an expert data collection programmer for spatial reasoning problems. Your task is to write a program that collects ALL potentially relevant 3D information needed to answer a spatial reasoning question. You do NOT need to determine the final answer — a reasoning agent will use your data to do that.

Your program must be put inside the tags <program></program>!!!
Now here is an API of methods, you will want to collect data in a logical and sequential manner, you don't need to import these functions in your program, you can use them directly:

------------------ API ------------------
{predef_signatures}
------------------ API ------------------

CRITICAL REQUIREMENTS:
1. Your program must end with creating a dictionary called "analysis_data" that contains all the processed information
2. DO NOT determine the final answer - that will be done by the reasoning agent
3. Include context information in the dictionary to help the reasoning agent understand what each data point means
4. Make sure to collect data systematically and thoroughly
5. Don't put pure numerical values into analysis_data, like camera extrinsics/intrinsics/world points this kind of information cannot be understood by the reasoning agent.

IMPORTANT: The following variables are already available at the start of your program (automatically computed):
- extrinsics: Camera extrinsic matrices for all images (np.array,shape: N, 3, 4)
- intrinsics: Camera intrinsic matrices for all images (np.array,shape: N, 3, 3)
- depth_maps: Depth maps for all images (np.array,shape: N, H, W)
- world_points: 3D world points from all images (np.array,shape: N*H*W, 3)
- ground_normal: Accurate ground plane normal vector (np.array,shape: 3,)
- ground_centroid: Accurate ground plane centroid vector (np.array,shape: 3,)

BASIC TIPS:
1) The total number of images is given in the question. Only process that many images. Use images[0], images[1], etc. Don't use wrong names like img1 or image_1.
2) You already have an initialized variable named "images" — no need to initialize it yourself.
3) For a camera pose, the forward direction is +Z, right is +X, down is +Y. If the question specifies -Z forward, use z_is_positive_forward=False!
4) Pass the pre-computed ground_normal to calibrate_directions/calculate_direction for better accuracy.
5) When using calibration functions, direction parameters must be "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest". Convert "front" to "north", "left/right" to "west/east" etc.
6) The shape of camera extrinsics is (N, 3, 4)!

OBJECT LOCALIZATION WORKFLOW:
7) First call find_obj(images, ["object_name"]) to find which image the object appears in.
   Then call get_object_3d_position(images[idx], extrinsics, intrinsics, depth_maps, idx, "object_name").
   This ensures detection in an image where the object is actually visible.
8) If get_object_3d_position returns None, try alternative descriptions (shorter names, synonyms, generic terms). Try every available image. Store the result even if None.
9) After detecting two objects, check if np.linalg.norm(pos_A - pos_B) < 0.1 — if so, detections may be aliased. Retry with more specific prompts or different images.
10) NEVER pass None positions to calibrate_directions, calibrate_from_vector, or calculate_direction. Always add None checks first.
11) After calibration, perform a sanity check: verify the known reference direction matches. Store the result.

QUESTION PATTERN TIPS:
12) "Standing at X, gazing at Y, where is Z?":
   - Locate X, Y, Z with get_object_3d_position (ALL THREE must succeed)
   - gaze_vector = Y_position - X_position
   - calibrate_from_vector(world_points, gaze_vector, X_position, "north", ground_normal)
   - calculate_direction(calibration_info, Z_position, X_position, ground_normal)
   - If any object returns None, try at least 5 alternative descriptions across all images. For abstract regions (bathroom/bedroom), detect a landmark instead (toilet/bed).
13) "Where is A relative to B from camera's perspective":
   - Use relative_object_position(cam_extrinsic, pos) for BOTH objects
   - Compute differences: diff_forward, diff_right, diff_up
   - NEVER do calibrate_directions(world_points, A_pos, B_pos, "north") then calculate_direction(cal, A_pos, B_pos) — this is CIRCULAR and always returns "north"!
   - For "where is X relative to camera", just use relative_object_position directly
14) "Where is X relative to camera at image N":
   - X may NOT be visible in image N! Use find_obj first, get 3D position from correct image, then relative_object_position(extrinsics[N], X_world_pos).
15) "From [person]'s perspective, where is [object]?":
   - Use situated_orientation_grounding to get the person's facing direction
   - calibrate_from_vector(world_points, facing_vector, person_pos, "north", ground_normal)
   - calculate_direction(calibration, object_pos, person_pos, ground_normal)
   - Do NOT use relative_object_position from camera — the person may face a different direction
16) "Object View Orientation" (which way is object facing):
   - Use situated_orientation_grounding to get facing direction
   - Store camera forward direction from extrinsics for comparison
   - Compute dot products between facing vector and camera axes
17) Rotation questions: compute ALL components (around_x, around_y, around_z) and store them. The axis with the LARGEST absolute angle is the dominant rotation.
18) Height/elevation comparisons: project onto ground_normal via dot product. Do NOT use raw Y-coordinates.
19) Motion questions: compute positions in world coordinates across frames. Small displacement (< 0.05) vs. camera motion may indicate a stationary object.
20) Abstract regions ("office area", "living room"): detect a concrete landmark instead ("desk", "sofa").
21) NEVER fabricate numerical measurements. Only report values computed by the API.
22) Note uncertainties or ambiguities in analysis_data.

Your program should end with:
analysis_data = {{......}}

Using the provided API, output a program inside the tags <program></program> to collect comprehensive data for answering the question.
Your program must be put inside the tags <program></program>!!!

<question>{question}</question>
"""

DATA_COLLECTION_PROMPT_TARGETED = """
You are an expert data collection programmer for spatial reasoning problems. The reasoning agent has analyzed the previously collected data and determined that additional information is needed.

Your task: Write a program to collect the SPECIFIC additional data requested below.

IMPORTANT: All previously computed variables are available in your program's namespace. This includes:
- extrinsics, intrinsics, depth_maps, world_points, ground_normal, ground_centroid
- images (list of input images)
- Any variables computed by your previous program (e.g., object positions, calibration data)
- The existing "analysis_data" dictionary from the previous run

You do NOT need to recompute anything that was already computed. Just collect the new information requested.

REQUEST FROM REASONING AGENT:
{request}

CURRENTLY COLLECTED INFORMATION:
{current_information_set}

ORIGINAL QUESTION:
{question}

Now here is the API of available methods (you can use them directly without importing):

------------------ API ------------------
{predef_signatures}
------------------ API ------------------

Your program must be put inside the tags <program></program>!!!

Write a program that:
1. Collects the specifically requested information using the available API and previously computed variables
2. Updates the analysis_data dictionary with the new information

Tips:
1) The existing analysis_data dict is available — add new entries directly.
2) All variables from previous execution are available (object positions, calibration results, etc.)
3) Focus ONLY on the requested data — do not re-collect existing data.
4) If the question specifies -Z forward, use z_is_positive_forward=False!
5) Direction parameters: "north", "south", "east", "west", etc. Convert "front" to "north", "left/right" to "west/east".
6) The shape of camera extrinsics is (N, 3, 4)!
7) If previous detection returned None, try alternative descriptions and different images.

Your program must be put inside the tags <program></program>!!!
"""


# Alias for backward compatibility
DATA_COLLECTION_PROMPT_MMSI = DATA_COLLECTION_PROMPT_INITIAL
