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


class ModelRequestError(RuntimeError):
    def __init__(self, message, quota_exhausted=False):
        super().__init__(message)
        self.quota_exhausted = quota_exhausted


class Generator:
    SUPPORTED_MODELS = [
        "gpt-4o-2024-11-20", "gemini-3.1-flash-lite",
        "gemini-3.5-flash", "gemini-3.1-pro-preview",
        # deprecated "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash","gpt-4o", "gpt-4o-mini",

    ]

    def __init__(self, model_name="gemini-3.1-flash-lite", temperature=0.7):
        self.temperature = temperature
        self.model_name = model_name
        self.custom_endpoint = bool(os.getenv("API_BASE_URL"))
        if model_name not in self.SUPPORTED_MODELS and not self.custom_endpoint:
            raise ValueError(f"Unsupported model: {model_name}")
        vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
        self.vertex = model_name.startswith("gemini-") and not self.custom_endpoint and (
            vertex in ("true", "1") or (not vertex and not os.getenv("GOOGLE_API_KEY")))
        if self.vertex:
            try:
                import google.auth
                from google import genai
            except ImportError as exc:
                raise ImportError("Vertex AI requires google-genai; install it with: pip install -U google-genai") from exc
            credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            project = os.getenv("GOOGLE_CLOUD_PROJECT") or project or getattr(credentials, "quota_project_id", None)
            if not project:
                raise ValueError("Set GOOGLE_CLOUD_PROJECT for Vertex AI")
            self.client = genai.Client(vertexai=True, project=project,
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), credentials=credentials,
                http_options={"timeout": 180000, "retry_options": {"attempts": 1}})
        else:
            api_key, base_url = self._resolve_api_config(model_name)
            self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=180)
        print(f"Using model: {model_name}")

    @staticmethod
    def _resolve_api_config(model_name):
        if os.getenv("API_BASE_URL"):
            key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set API_KEY or OPENAI_API_KEY for API_BASE_URL")
            return key, os.environ["API_BASE_URL"]
        if model_name.startswith("gpt-") and os.getenv("OPENAI_API_KEY"):
            return os.environ["OPENAI_API_KEY"], "https://api.openai.com/v1"
        if model_name.startswith("gemini-") and os.getenv("GOOGLE_API_KEY"):
            return os.environ["GOOGLE_API_KEY"], "https://generativelanguage.googleapis.com/v1beta/openai/"
        raise ValueError("Set OPENAI_API_KEY, GOOGLE_API_KEY, or API_BASE_URL with API_KEY; Vertex AI uses ADC")

    @property
    def thinking_level(self):
        if self.custom_endpoint:
            return None
        if self.model_name == "gemini-3.1-pro-preview":
            return "HIGH"
        if self.model_name in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
            return "LOW"
        return None

    @property
    def max_output_tokens(self):
        if self.custom_endpoint or not self.model_name.startswith("gemini-"):
            return None
        return 32768 if self.thinking_level == "HIGH" else 8192

    def _vertex_generate(self, messages, max_output_tokens, response_format):
        from google.genai import types
        options = {}
        if response_format is not None:
            if response_format.get("type") != "json_schema":
                raise ValueError("Gemini structured output requires a JSON schema")
            options.update(response_mime_type="application/json",
                           response_json_schema=response_format["json_schema"]["schema"])
        if self.thinking_level is not None:
            options["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        contents, instructions = [], []
        for message in messages:
            role, content = message["role"], message["content"]
            items = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if role == "system":
                if any(item["type"] != "text" for item in items):
                    raise ValueError("Gemini system instructions must contain only text")
                instructions.extend(item["text"] for item in items)
                continue
            if role not in ("user", "assistant"):
                raise ValueError(f"Unsupported Gemini message role: {role}")
            parts = []
            for item in items:
                if item["type"] == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError("Vertex AI images must use base64 image data URLs")
                    header, encoded = url.split(",", 1)
                    parts.append(types.Part.from_bytes(data=base64.b64decode(encoded, validate=True),
                        mime_type=header.split(":", 1)[1].split(";")[0]))
                else:
                    raise ValueError(f"Unsupported content type: {item['type']}")
            contents.append(types.Content(role="model" if role == "assistant" else "user", parts=parts))
        response = self.client.models.generate_content(model=self.model_name, contents=contents,
            config=types.GenerateContentConfig(system_instruction="\n\n".join(instructions) or None,
                temperature=self.temperature, max_output_tokens=max_output_tokens, **options))
        candidate = response.candidates[0] if response.candidates else None
        reason = getattr(candidate, "finish_reason", None)
        reason = getattr(reason, "value", reason)
        blocked = reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY")
        blocked = blocked or bool(getattr(response.prompt_feedback, "block_reason", None))
        return response.text or "", response.model_version, reason, blocked

    def remove_substring(self, output, substring):
        return output.replace(substring, "")

    def generate(self, prompt, messages=None, _retry_count=0, response_format=None):
        import signal
        import threading
        timer = None
        if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():
            timer = signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            return self._generate_messages(prompt, messages, _retry_count, response_format)
        finally:
            if timer is not None and timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *timer)

    def _generate_messages(self, prompt, messages, retry_count, response_format):
        import copy
        import random
        from openai import APIConnectionError
        from httpx import TransportError
        if type(retry_count) is not int or not 0 <= retry_count < 6:
            raise ValueError("_retry_count must be an integer between 0 and 5")
        messages = copy.deepcopy(messages) if messages else [{"role": "user", "content": prompt}]
        budget = self.max_output_tokens
        for attempt in range(retry_count, 6):
            try:
                if self.vertex:
                    output, actual_model, reason, blocked = self._vertex_generate(messages, budget, response_format)
                else:
                    options = {"response_format": response_format} if response_format is not None else {}
                    if budget is not None:
                        options["max_tokens"] = budget
                    if self.thinking_level is not None:
                        options["extra_body"] = {"reasoning_effort": self.thinking_level.lower()}
                    response = self.client.chat.completions.create(
                        model=self.model_name, messages=messages, temperature=self.temperature, **options)
                    choice = response.choices[0] if response.choices else None
                    output = choice.message.content or "" if choice else ""
                    actual_model = response.model
                    reason = choice.finish_reason if choice else None
                    blocked = reason == "content_filter" or bool(getattr(choice.message, "refusal", None) if choice else None)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                quota = any(term in str(exc).lower() for term in ("insufficient_quota", "credit_balance_exhausted"))
                transient = status in (408, 429, 500, 502, 503, 504) or isinstance(exc, (APIConnectionError, TransportError, TimeoutError, ConnectionError))
                if quota or not transient or attempt + 1 >= (6 if status == 429 else 4):
                    raise ModelRequestError(str(exc), quota_exhausted=quota) from exc
                failure = f"{type(exc).__name__}, status={status}"
            else:
                versioned_alias = actual_model and actual_model.startswith(self.model_name + "-") and actual_model[len(self.model_name) + 1:len(self.model_name) + 2].isdigit()
                if not self.custom_endpoint and actual_model and not (actual_model == self.model_name or versioned_alias):
                    raise ModelRequestError(f"Model mismatch: requested {self.model_name}, received {actual_model}")
                if blocked:
                    raise ModelRequestError(f"Model response blocked: {reason}")
                if reason in ("MAX_TOKENS", "length"):
                    limit = 8192 if self.model_name.startswith("gemini-2.0-") else 65536
                    if budget is None or budget >= limit:
                        raise ModelRequestError("Model response truncated at its output limit")
                    budget = min(budget * 2, limit)
                    failure = f"Model response truncated; max_output_tokens={budget}"
                elif output.strip():
                    output = output.strip().replace("```python", "").replace("```json", "").replace("```", "")
                    if not output.strip():
                        failure = "Model returned no content after formatting removal"
                    else:
                        messages.append({"role": "assistant", "content": output})
                        return output, messages
                else:
                    failure = "Model returned no text"
                if attempt + 1 >= 4:
                    raise ModelRequestError(failure)
            delay = min(10 * 2 ** attempt, 90) + random.uniform(0, 5)
            print(f"API retry {attempt + 1}: {failure}, wait={delay:.1f}s")
            time.sleep(delay)


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
        "Local program execution exceeded its time budget. Inspect the traceback for slow loops or expensive computation."
    )


def remove_python_text(output):
    substring = "```python"
    if substring in output:
        output = output.replace(substring, "")

    substring = "```"
    if substring in output:
        output = output.replace(substring, "")

    return output
