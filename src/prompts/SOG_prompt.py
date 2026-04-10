COARSE_PROMPT = """
Context:
You are analyzing a scene with four directional vectors labeled 0, 1, 2, and 3. You will receive TWO SEPARATE IMAGES of the same scene: the FIRST IMAGE shows the original camera perspective, and the SECOND IMAGE shows an elevated bird's-eye perspective for better clarity. Both images show the same arrows at the same 3D location. Your task is to identify the single vector that most closely represents the described orientation.

Vector Rules:
1. Horizontal Plane: These vectors exist on a single horizontal plane, representing directions parallel to the ground (like a compass). Ignore any "up" or "down" components.
2. Cross Shape: The four vectors are arranged in a cross shape, with a 90-degree angle between each adjacent vector (e.g., between 0 and 1, 1 and 2, etc.).
3. Dual View Analysis: Use BOTH images to understand the true 3D direction of each arrow. The elevated view (second image) provides clearer perspective on horizontal directions.

Your Goal:
Based on the orientation description provided, select the ONE arrow that best represents the direction. The chosen arrow's direction should be the *closest* match to the target direction on the horizontal plane.

Key Instructions & Constraints:
- Focus on Line of Sight: For actions involving a person, your answer should represent their most likely line of sight or the direction they are primarily facing to perform the action.
- Horizontal Projection Only: If a described behavior includes any upward or downward motion (e.g., "looking up at the ceiling," "picking something up from the floor"), you must ignore the vertical component. Project the action onto the horizontal plane, as if you were looking at the scene from a top-down perspective.
- 3D Perspective is Crucial: The vectors are in 3D space but projected onto a 2D image. You must analyze their appearance (e.g., length, angle) to infer their true 3D direction. Pay special attention to whether a vector points towards you (out of the image) or away from you (into the image). Foreshortening (an object appearing shorter when pointing towards/away from the camera) is a key clue.

Reasoning Process:
You must first provide a step-by-step reasoning process before giving the final answer. Follow this structure:
1. Analyze the camera's position and orientation: Based on objects you observe in the FIRST image (original perspective), determine the camera's position and orientation.("inside of the room", "outside of the room", "facing the door", "facing the window", etc.)
2. Analyze Target Direction: Briefly explain what the `orientation_description` means in terms of direction (e.g., "moving forward into the scene," "looking to the left").
3. Analyze Image Context & Arrow Orientations: Describe the scene and then determine the real-world direction of *each* of the four arrows (0, 1, 2, 3) using BOTH images:
    * Use the FIRST image (original perspective) to understand the spatial context and 3D foreshortening effects
    * Use the SECOND image (elevated perspective) to clearly see horizontal directions without perspective distortion
    * For each arrow, state its direction clearly (e.g., "Arrow 0 points to the left", "Arrow 1 points away from camera")
4. Compare and Select: Compare the target direction from step 2 with the arrow orientations from step 3. Identify which arrow is the closest match and justify your choice using evidence from both images.

Example of a High-Quality Reasoning Process:
<example>
Description: "going out of the living room through the door"

Reasoning:
1. Analyze the camera's position and orientation: There are sofas near the camera, so the side of camera is located in the room.
2. Analyze Target Direction: The description "going out of the room through the door" implies a direction of movement starting from the camera's viewpoint, moving forward through the doorway, and away from the current position. This is an "away from the viewer" or "into the scene" direction.
3. Analyze Image Context & Arrow Orientations: The image displays an indoor scene, looking towards a doorway where the four directional arrows are placed.
    * Arrow 0: Points horizontally to the left, parallel to the plane of the door.
    * Arrow 1: This arrow is visibly shorter than arrows 0 and 2. This foreshortening effect strongly indicates that it is pointing either directly away from or towards the camera along the line of sight. Given the context of the door, it points directly *into* the doorway, representing the direction "away from the camera".
    * Arrow 2: Points horizontally to the right, opposite to arrow 0.
    * Arrow 3: This arrow is longer than arrow 1 and points towards the bottom of the image. It represents the direction "towards the camera" or "out of the scene".
3. Compare and Select: The target direction is "going out," which perfectly aligns with the "away from the camera" direction. Based on the perspective analysis, Arrow 1 is the one that points away from the camera and into the doorway. Therefore, Arrow 1 is the best match.
</example>

Final Output Format:
First, output your entire reasoning process within <Reasoning> tags. Then, provide the single number of the chosen arrow within <answer> tags.

Example: <Reasoning>...your detailed reasoning here...</Reasoning><answer>1</answer>

Original orientation description: "{orientation_description}"
"""

FINE_PROMPT = """
Context:
You are analyzing a scene with five directional vectors. You will receive TWO SEPARATE IMAGES of the same scene: the FIRST IMAGE shows the original camera perspective, and the SECOND IMAGE shows an elevated bird's-eye perspective for better clarity. Both images show the same arrows at the same 3D location. Your task is to select the single vector that most precisely represents the described orientation.

Vector Rules:
1. Horizontal Plane: These vectors exist on a single horizontal plane, representing directions parallel to the ground (like a compass). Ignore any "up" or "down" components.
2. Shape: The five vectors are arranged in a plane parallel to the ground, with a 22.5-degree angle between each adjacent vector (e.g., between 0 and 1, 1 and 2, etc.).
3. Dual View Analysis: Use BOTH images to understand the precise 3D direction of each arrow. The elevated view (second image) provides clearer perspective on subtle angular differences.

Vector Definitions:
The five vectors are arranged in a fan shape, representing fine-grained adjustments around a central direction. They are labeled as follows:
- `2`: The Center vector, representing the main direction.
- `1`: Rotated 22.5 degrees Clockwise from the Center vector.
- `0`: Rotated 45 degrees Clockwise from the Center vector.
- `3`: Rotated 22.5 degrees Counter-Clockwise from the Center vector.
- `4`: Rotated 45 degrees Counter-Clockwise from the Center vector.

Your Goal:
Based on the orientation description, select the ONE vector that is the absolute best match. The difference between these vectors is small, so you must pay close attention to subtle details.

Key Instructions & Constraints:
- Focus on Nuance: This is a precision task. Re-read the description and re-examine the image for subtle clues. Does the description imply a slight angle (e.g., "towards the corner," "just to the left of the window")? Does a person's body posture or gaze suggest a direction that isn't perfectly straight?
- Horizontal Projection Only: Ignore all vertical (up/down) components. Your decision must be based on the direction projected onto the horizontal ground plane.
- 3D Perspective is Crucial: Analyze the scene's perspective to understand what "Clockwise" and "Counter-Clockwise" mean. For example, a clockwise rotation might mean turning more "into the scene" or more "towards the right," depending on the camera angle.

Reasoning Process:
You must provide a step-by-step reasoning process before your final answer. Follow this structure:
1. Analyze the camera's position and orientation: Based on objects you observe in the FIRST image (original perspective), determine the camera's position and orientation.("inside of the room", "outside of the room", "facing the door", "facing the window", etc.)
2. Analyze Target Direction for Nuance: First, determine the general direction from the description. Then, identify the *subtle, precise details* that refine this direction. This often involves understanding the spatial context of the scene.
3. Analyze Image & Vector Orientations: Describe how the five vectors are oriented within the scene's 3D space using BOTH images:
    * Use the FIRST image (original perspective) to understand the spatial context
    * Use the SECOND image (elevated perspective) to clearly see subtle angular differences between vectors
    * Clearly state what direction each label corresponds to (e.g., "2 points straight ahead. 1 points straight ahead and slightly right...")
4. Compare and Select Precisely: Compare the nuanced target direction from step 2 with the vector orientations from step 3. Justify why your chosen vector is a better fit than its immediate neighbors using evidence from both images (e.g., "Why is 3 the best choice, and not 2 or 4?").

Example of a High-Quality Reasoning Process:
<example>
Description: "entering the bathroom"

Reasoning:
1. Analyze the camera's position and orientation: The image shows a doorway, and through it, I can see a sink and a toilet, which are typical bathroom fixtures. The area where the camera is located appears to be a hallway or bedroom. This spatial analysis confirms that the viewpoint is **outside** the bathroom, looking in. Therefore, the action of "entering" means moving from the current location (outside) *through the doorway* and *into* the bathroom. This translates to a primary direction that is **away from the camera**. Looking closer at the layout inside, the path seems to angle slightly to the left to navigate the space. So, the precise target direction is "away from the camera and slightly to the left".
2. Analyze Target Direction for Nuance: The action is "entering the bathroom". To determine the vector for this, I must first locate the bathroom door and then understand the camera's viewpoint. 
3. Analyze Image & Vector Orientations: The five vectors originate near the doorway. Based on the perspective:
    * `2` (Center): Points straight forward, directly away from the camera and into the bathroom.
    * `1` (Rotated Clockwise): Points away from the camera and slightly to the right.
    * `0` (Rotated further Clockwise): Points away from the camera and significantly to the right.
    * `3` (Rotated Counter-Clockwise): Points away from the camera and slightly to the left.
    * `4` (Rotated further Counter-Clockwise): Points away from the camera and significantly to the left.
3. Compare and Select Precisely: The nuanced target direction is "away from the camera and slightly to the left". Vector `2` correctly identifies the "away from camera" component but is too direct. Vectors `1` and `0` are incorrect as they point to the right. Vector `4` represents too sharp of a turn to the left upon entering. Vector `3` perfectly captures the intended path: moving forward into the bathroom while veering slightly to the left. Thus, it is the most accurate choice.
</example>

Final Output Format:
First, output your reasoning within <Reasoning> tags. Then, provide the label of the chosen vector (`0`, `1`, `2`, `3`, or `4`) within <answer> tags.

Example: <Reasoning>...your detailed reasoning here...</Reasoning><answer>3</answer>

Original orientation description: "{orientation_description}"
"""

IDENTIFY_KEY_OBJECT_PROMPT = """
Given this orientation description: "{orientation_description}"

What is the main object or landmark that this orientation is related to? 
For example:
- "direction going up the stairs" → "stairs"
- "direction exiting the bathtub" → "bathtub"  
- "direction entering the room" → "door" or "doorway"
- "direction facing the window" → "window"
- "the facing of the laptop" → "laptop"
- "the direction of the plant facing the wall" → "plant&wall"
- "entering the bathroom" → "bathroom door"
Respond with ONLY the object name, nothing else. Use simple, specific terms that would be good for object detection. The answer must be included in <answer>object_name</answer> tags. e.g. <answer>laptop</answer>
"""

UNIFIED_OBJECT_SELECTION_PROMPT = """
Context:
You are helping with a spatial orientation task. Given an orientation description and multiple images of a scene, you need to:
1. Identify the most relevant object/landmark for the described orientation
2. Select which image shows this object most completely and clearly

Your selections will be used to place directional arrows around the chosen object in the selected image, so it's crucial that:
- The object is clearly visible and unobstructed
- The object is large enough to serve as a reference point for direction arrows
- The viewing angle allows for clear spatial understanding

Not everytime there is a perfect match, you should select the best match according to the requirement.
Task:
Given the orientation description: "{orientation_description}"
The images id are: {image_ids}
Analyze the provided images (numbered 0, 1, 2, ...) and determine:
1. What object or landmark is most relevant to this orientation description
2. Which image shows this object most completely, clearly, and from the best angle for placing directional arrows

Examples:

<example>
Orientation: "going out through the front door"
Images: [Image 0: Shows a living room with partial door visible at edge, Image 1: Shows the front door fully centered with clear surrounding area, Image 2: Shows a kitchen view]
Analysis:
- Target Object: "door" (the front door is the key landmark for the "going out" action)
- Best Image: Image 1 (shows the door completely, centrally positioned, with clear spatial context for arrow placement)
- Reasoning: Image 1 provides the clearest view of the door with sufficient surrounding space to place directional arrows that would help determine the "going out" direction
</example>

<example>
Orientation: "direction facing the TV while sitting on the couch"
Images: [Image 0: Shows partial couch from side, Image 1: Shows TV mounted on wall but couch not visible, Image 2: Shows both couch and TV clearly with good perspective]
Analysis:
- Target Object: "couch" (the couch is where the person sits, making it the reference point for the "facing TV" orientation)
- Best Image: Image 2 (shows both couch and TV with clear spatial relationship and good viewing angle)
- Reasoning: The couch is the anchor point for this orientation. Image 2 shows the couch clearly with its spatial relationship to the TV, perfect for placing arrows to indicate the "facing TV" direction
</example>

Instructions:
1. Carefully examine each image
2. Identify the most relevant object for the orientation description
3. Determine which image shows this object most clearly and completely
4. Consider how directional arrows would be placed around this object

Output Format:
First, reason. Then, provide your answer in the following JSON format within <answer> tags:
<Reasoning>...</Reasoning>
<answer>
{{  
    "target_object": "object_name",
    "best_image_id": number
}}
</answer>

Orientation Description: "{orientation_description}"
Not everytime there is a perfect match, you should select the best match according to the requirement.
"""