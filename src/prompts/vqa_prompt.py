VQA_PROMPT_CLEVR = """You will be shown an image with a bounding box and asked to answer a question based on the object inside the bounding box.
Available sizes are {{small, large}}, available shapes are {{cube, sphere, cylinder}}, available material types are {{rubber, metal}}, available colors are {{gray, blue, brown, yellow, red, green, purple, cyan}}. Please DO NOT answer with attributes other than those that I specified!
If you think the right answer isn't one of the available attributes, choose the available attribute that is most similar!
Answer this question using the attributes as reference based on the object in the bounding box and put your answer in between <answer></answer> tags: {question}"""

VQA_PROMPT_GQA = """
You will be shown an image with a bounding box and asked to answer a question based on the object inside or around the bounding box.
Your answer should be a single word.
The answer should be simple and generic.
If the answer is not generic I will kill you.
If asked about people, do not just say "person".
Never respond with "unknown" or "none", always give a plausible answer.
Put your answer in between <answer></answer> tags: 
Question:{question}"""

VQA_PROMPT_GQA_HOLISTIC = """
Answer this question. 
Your answer should be a single word.
If the answer is based on an object, the answer should be simple and generic.
If the answer is not generic I will kill you.
If asked about people, do not just say "person".
Never respond with "unknown" or "none", always give a plausible answer.
Put your answer in between <answer></answer> tags: 
Question:{question}"""

VQA_PROMPT_MMSI = """
You will be shown an image and asked to answer a question. 
Your answer should be a single word, like 'yes' or 'no'.
Never respond with "unknown" or "none", always give a plausible answer.
Put your answer in between <answer></answer> tags.
Question: {question}"""

FIND_OBJ_PROMPT_MMSI = """
You will be shown a series of images.
For each of the following objects, please identify the image ID where the object appears most clearly and completely, with minimal obstruction.
The objects to find are:
{object_prompts}

The available image IDs are: {image_ids}.
If the object is not in any of the images, exclude it from the answer.
Your answer must be a JSON dictionary where keys are the object descriptions and values are the corresponding image IDs. The dictionary must be wrapped in <answer> tags.
For example:
<answer>
{{
  "white rug": 1,
  "blue car": 3
}}
</answer>
"""
