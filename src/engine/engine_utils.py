import numpy as np
from openai import OpenAI
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import time
import torch
import os

def correct_indentation(code_str):
    lines = code_str.split("\n")
    tabbed_lines = ["\t" + line for line in lines]
    tabbed_text = "\n".join(tabbed_lines)
    return tabbed_text


def replace_tabs_with_spaces(code_str):
    return code_str.replace("\t", "    ")


def untab(text):
    lines = text.split("\n")
    untabbed_lines = []
    for line in lines:
        if line.startswith("\t"):
            untabbed_lines.append(line[1:])
        elif line.startswith("    "):
            untabbed_lines.append(line[4:])
        else:
            untabbed_lines.append(line)
    untabbed_text = "\n".join(untabbed_lines)
    return untabbed_text


def get_methods_from_json(api_json):
    methods = []
    namespace = {}
    for method_info in api_json:

        signature = method_info["signature"]
        if "def " in method_info["implementation"]:
            full_method = method_info["implementation"]
        else:
            full_method = signature + "\n" + method_info["implementation"]
        methods.append(full_method)

    return methods, namespace


class Generator:
    SUPPORTED_MODELS = [ # you can add other models
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    def __init__(self, model_name="gemini-3.1-flash-lite-preview", temperature=0.7):
        self.temperature = temperature
        self.model_name = model_name

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Supported: {self.SUPPORTED_MODELS}")

        api_key, base_url = self._resolve_api_config(model_name)
        print(f"Using model: {model_name} via {base_url}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _resolve_api_config(model_name):
        """Resolve API key and base URL from environment variables.

        Priority order:
        1. API_BASE_URL + API_KEY (explicit override for custom endpoints)
        2. OPENAI_API_KEY (native OpenAI, for gpt-* models)
        3. GOOGLE_API_KEY via OpenAI-compatible endpoint (for gemini-* models)
        """
        # Check for explicit override first
        if os.getenv("API_BASE_URL"):
            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            return api_key, os.getenv("API_BASE_URL")

        # Native OpenAI
        if model_name.startswith("gpt-") and os.getenv("OPENAI_API_KEY"):
            return os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1"

        # Google Gemini via OpenAI-compatible endpoint
        if model_name.startswith("gemini-") and os.getenv("GOOGLE_API_KEY"):
            return os.getenv("GOOGLE_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/"

        raise ValueError(
            "No API key found. Set one of the following environment variables:\n"
            "  - OPENAI_API_KEY (for GPT models via OpenAI)\n"
            "  - GOOGLE_API_KEY (for Gemini models via Google)\n"
            "  - API_BASE_URL + API_KEY (for custom OpenAI-compatible endpoints)"
        )

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def generate(self, prompt, messages=None, _retry_count=0):
        max_retries = 5
        new_messages = None

        if not messages:
            messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            if _retry_count >= max_retries:
                raise RuntimeError(f"API call failed after {max_retries} retries. Last error: {e}")
            wait_time = min(60 * (2 ** _retry_count), 300)
            print(f"Error: {e}, retrying in {wait_time}s (attempt {_retry_count + 1}/{max_retries})")
            time.sleep(wait_time)
            return self.generate(prompt, messages, _retry_count=_retry_count + 1)
        new_messages = messages
        result = response.choices[0].message.content.lstrip("\n").rstrip("\n")
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```json")
        result = self.remove_substring(result, "```")
        new_messages.append(
            {
                "role": response.choices[0].message.role,
                "content": result,
            }
        )
        return result, new_messages

if __name__ == "__main__":
    generator = Generator(model_name="gemini-3.1-flash-lite-preview")
    result, messages = generator.generate("Tell me a joke")
    print(result)
    print(messages)
    

def docstring_from_json(json_file):
    with open(json_file, "r") as file:
        api_data = json.load(file)

    docstring = ""
    for module in api_data.get("modules", []):
        docstring += f'"""\n'
        docstring += f"{module['description']}\n\n"
        if module["arguments"]:
            docstring += "Args:\n"
            for arg in module["arguments"]:
                docstring += (
                    f"    {arg['name']} ({arg['type']}): {arg['description']}\n"
                )
        if "returns" in module:
            docstring += f"\nReturns:\n"
            docstring += f"    {module['returns']['type']}: {module['returns']['description']}\n\"\"\""
        docstring += f"\n{module['name']}("
        args = [arg["name"] for arg in module["arguments"]]
        docstring += ", ".join(args) + ")\n\n"

    return docstring.strip()

def depth_to_grayscale(depth_map):
    # Ensure depth_map is a NumPy array of type float (if not already)
    depth_map = np.array(depth_map, dtype=np.float32)

    # Get the minimum and maximum depth values
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    
    # Avoid division by zero if the image is constant
    if d_max - d_min == 0:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - d_min) / (d_max - d_min)
    
    # Scale to 0-255 and convert to unsigned 8-bit integer
    grayscale = (normalized * 255).astype(np.uint8)
    
    return grayscale


def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1


def dotted_image(img, points):
    # Scale dot size based on image width
    if isinstance(img, np.ndarray):
        img_width = img.shape[1]
        np_img = img.copy()
        img = Image.fromarray(np_img)
        if img.mode == 'F':
            img = depth_to_grayscale(np_img)
            img = Image.fromarray(img)
            img = img.convert('RGB')
    else:
        img_width = img.size[0]

    
    dot_size = int(img_width * 0.02) # 2% of image width
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for pt in points:
        x = pt[0]
        y = pt[1]

        draw.ellipse(
            (x - dot_size, y - dot_size, x + dot_size, y + dot_size),
            fill="red",
            outline="black",
        )
    return img1


def html_embed_image(img, size=300):
    img = img.copy()
    img.thumbnail((size, size))
    with BytesIO() as buffer:
        if img.mode == 'F':
            img = img.convert('RGB')
        img.save(buffer, "png")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/png;base64,{base64_img}">'
    )
    

class TimeoutException(Exception):
    pass


def set_devices():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def timeout_handler(signum, frame):
    raise TimeoutException(
        "The script took too long to run. There is likely a Recursion Error. Ensure that you are not calling a method with infinite recursion."
    )


def remove_python_text(output):
    substring = "```python"
    if substring in output:
        output = output.replace(substring, "")

    substring = "```"
    if substring in output:
        output = output.replace(substring, "")

    return output
