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

def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1

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
    def __init__(self, model_name="gpt-4o", temperature=0.7, thinking_budget=0, api_key_path="./api.key"):
        self.temperature = temperature
        self.model_name = model_name
        self.api_key_path = api_key_path
        self.thinking_budget = thinking_budget
        print(f"Using model: {model_name}", 'temperature', temperature, 'thinking_budget', thinking_budget)
        if model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-pro"]:
            self.client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        elif model_name in ["gpt-4o", "gpt-4o-mini"]:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Model {model_name} not supported")
        

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def generate(self, prompt, messages=None):
        new_messages = None

        if not messages:
            messages = [{"role": "user", "content": prompt}]
        try:
            if self.model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite-preview-06-17"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    extra_body={ # for Gemini thinking models
                        'extra_body': {
                            "google": {
                            "thinking_config": {
                                "thinking_budget": self.thinking_budget,
                                "include_thoughts": True
                            }
                            }
                        }
                    }
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
        except Exception as e:
            print(f"Error: {e}, sleeping for 60 seconds")
            time.sleep(60)
            return self.generate(prompt, messages)
        new_messages = messages
        result = response.choices[0].message.content.lstrip("\n").rstrip("\n")
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```json")
        result = self.remove_substring(result, "json")
        result = self.remove_substring(result, "```")
        new_messages.append(
            {
                "role": response.choices[0].message.role,
                "content": result,
            }
        )
        return result, new_messages

if __name__ == "__main__":
    generator = Generator(model_name="gpt-4o-mini")
    result, messages = generator.generate("Tell me a joke")
    print(result)
    print(messages)
    


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
