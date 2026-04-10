import os
import sys
import json
import base64
import io
import re
from tqdm import tqdm
from PIL import Image

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from engine.engine_utils import Generator
from prompts.analysis_prompt import REASONING_AGENT_PROMPT


class Agent:
    def __init__(
        self,
        model_name="gemini-3.1-flash-lite-preview",
        write_results=True,
        dataset="mmsi-bench",
    ):
        self.generator = Generator(model_name)
        self.write_results = write_results
        self.dataset = dataset


class ProgramAgent(Agent):
    def __init__(
        self, api_agent=None, model_name="gemini-3.1-flash-lite-preview", write_results=True, dataset="mmsi-bench", predef_signatures=None, prompt_mode="text"
    ):
        super().__init__(model_name, write_results, dataset=dataset)
        self.api_agent = api_agent
        self.programs = []
        self.predef_signatures = predef_signatures
        self.prompt_mode = prompt_mode

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def __call__(self, question, prompt, images_folder_path=None):

        prompt_text = prompt.format(predef_signatures=self.predef_signatures, question=question["question"])

        if self.prompt_mode == "with-image":
            messages = [{"type": "text", "text": prompt_text}]
            for image_path in question["image_paths"]:
                prompt_text += "Images of the question are given to help you solve the question better."
                full_image_path = os.path.join(images_folder_path, image_path)
                image = Image.open(full_image_path)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "detail": "low"
                    }
                )

            output, messages = self.generator.generate("", messages=[{"role": "user", "content": messages}])
        else:
            output, messages = self.generator.generate(prompt_text)

        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
        self.programs.append(
            {
                "image_paths": question["image_paths"],
                "question_index": question["id"],
                "program": program,
                "prompt": prompt_text,
                "output": output,
                "messages": messages,
                "model_name": self.generator.model_name,
            }
        )

        return program, output

    def generate_targeted(self, question, prompt_template, request_text, current_info_set, images_folder_path=None):
        """Generate a targeted data collection program based on a reasoning agent's request.

        Args:
            question: Question data dict
            prompt_template: The targeted prompt template
            request_text: The specific request from the reasoning agent
            current_info_set: The current information set as formatted text
            images_folder_path: Path to images

        Returns:
            tuple: (program_code_list, raw_output)
        """
        prompt_text = prompt_template.format(
            predef_signatures=self.predef_signatures,
            question=question["question"],
            request=request_text,
            current_information_set=current_info_set,
        )

        if self.prompt_mode == "with-image":
            messages = [{"type": "text", "text": prompt_text}]
            for image_path in question["image_paths"]:
                full_image_path = os.path.join(images_folder_path, image_path)
                image = Image.open(full_image_path)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "detail": "low"
                    }
                )
            output, messages = self.generator.generate("", messages=[{"role": "user", "content": messages}])
        else:
            output, messages = self.generator.generate(prompt_text)

        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
        self.programs.append(
            {
                "image_paths": question["image_paths"],
                "question_index": question["id"],
                "program": program,
                "prompt": prompt_text,
                "output": output,
                "messages": messages,
                "model_name": self.generator.model_name,
                "iteration_type": "targeted",
                "request": request_text,
            }
        )

        return program, output

    def get_programs(
        self,
        questions_data,
        images_folder_path,
        results_folder_path,
        prompt,
    ):

        folder_name = "program_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)

        for question_data in tqdm(questions_data):
            html_path = os.path.join(
                    results_folder_path,
                    f"question_{question_data['id']}.html",
                )

            question = question_data["question"]
            program, output = self(question_data, prompt, images_folder_path=images_folder_path)

            if self.write_results:
                with open(html_path, "wb+") as file:

                    # Write question
                    file.write((f"<h1>{question}</h1>\n").encode("utf-8"))

                    # Handle multiple images
                    if "image_paths" in question_data:
                        for i, image_path in enumerate(question_data["image_paths"]):
                            image = Image.open(
                                os.path.join(images_folder_path, image_path)
                            )
                            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                            rgb_image = image.convert("RGB")
                            image_io = io.BytesIO()
                            rgb_image.save(image_io, format="PNG")
                            image_bytes = base64.b64encode(image_io.getvalue()).decode("ascii")

                            file.write((f"<h2>Image {i+1}</h2>\n").encode("utf-8"))
                            file.write(
                                (f"<img src='data:image/jpeg;base64,{image_bytes}'>\n").encode(
                                    "utf-8"
                                )
                            )
                    elif "image_filename" in question_data:
                        image = Image.open(
                            os.path.join(
                                images_folder_path, question_data["image_filename"]
                            )
                        )
                        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                        rgb_image = image.convert("RGB")
                        image_io = io.BytesIO()
                        rgb_image.save(image_io, format="PNG")
                        image_bytes = base64.b64encode(image_io.getvalue()).decode("ascii")

                        file.write(
                            (f"<img src='data:image/jpeg;base64,{image_bytes}'>\n").encode(
                                "utf-8"
                            )
                        )


                    file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{prompt.format(predef_signatures=self.predef_signatures, question=question_data['question'])}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>LLM Outputs</h1>\n").encode("utf-8"))
                    if isinstance(output, list):
                        for out in output:
                            file.write(
                                (f"<code>{out}</code>\n".replace("\n", "<br>")).encode(
                                    "utf-8"
                                )
                            )
                    else:
                        file.write(
                            (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )

                    file.write((f"<h1>Program</h1>\n").encode("utf-8"))
                    if len(program) > 0:
                        file.write(
                            (f"<code>{program[0]}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )
                    else:
                        file.write((f"<p>No program found</p>").encode("utf-8"))

                    file.close()


        return self.programs


class AnalysisAgent(Agent):
    def __init__(
        self, model_name="gemini-3.1-flash-lite-preview", write_results=True, dataset="mmsi-bench", max_retries=3
    ):
        super().__init__(model_name, write_results, dataset=dataset)
        self.analyses = []
        self.max_retries = max_retries

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def curate_and_decide(self, question_data, program_code, tools_definitions, information_set_text, iteration=0, force_decide=False):
        """Run the reasoning agent's curation and decision process.

        The reasoning agent examines the information set, curates it (keeps relevant
        items, discards irrelevant ones), and either decides on an answer or requests
        additional information from the perception agent.

        Args:
            question_data: Question dict with 'question' key
            program_code: The perception agent's code (for context)
            tools_definitions: Available tool API signatures
            information_set_text: Formatted text of the current information set
            iteration: Current iteration number
            force_decide: If True, force RA to make a decision (no Request allowed)

        Returns:
            dict with keys:
                - "type": "decide" or "request"
                - "answer": str or None (only for "decide")
                - "request_text": str or None (only for "request")
                - "full_output": str (raw LLM output)
                - "curated_set": str (extracted curated set text)
        """
        prompt_text = REASONING_AGENT_PROMPT.format(
            question=question_data["question"],
            program_code=program_code,
            tools_definitions=tools_definitions,
            information_set=information_set_text,
        )

        if iteration > 0 and not force_decide:
            prompt_text += f"\n\nNote: This is iteration {iteration + 1}. Previous requests have already been fulfilled. Carefully check if the information is now sufficient."

        if force_decide:
            prompt_text += """

CRITICAL: This is the FINAL iteration. You have NO more chances to request data.
You MUST output <Decide> with your best answer NOW. Do NOT output <Request>.
Even if the information is incomplete, use whatever data is available to make your best judgment.
You MUST choose one of A, B, C, or D. Analyze each option and pick the most likely one.
Output format: <Decide>...reasoning...<answer>X</answer></Decide>"""

        output, messages = self.generator.generate(prompt_text)

        result = self._parse_decision(output)
        result["full_output"] = output
        result["messages"] = messages
        result["iteration"] = iteration

        self.analyses.append({
            "question_id": question_data.get("id", "unknown"),
            "question": question_data["question"],
            "iteration": iteration,
            "decision_type": result["type"],
            "answer": result.get("answer"),
            "request_text": result.get("request_text"),
            "full_output": output,
            "model_name": self.generator.model_name,
        })

        return result

    def _parse_decision(self, output):
        """Parse the reasoning agent's output to determine decision type.

        Returns:
            dict with "type" ("decide" or "request"), "answer", "request_text", "curated_set"
        """
        result = {
            "type": None,
            "answer": None,
            "request_text": None,
            "curated_set": "",
        }

        # Extract curated set if present
        curated_match = re.search(r"CURATED SET:(.*?)(?:DECISION:|$)", output, re.DOTALL | re.IGNORECASE)
        if curated_match:
            result["curated_set"] = curated_match.group(1).strip()

        # Check for <Decide> tag
        decide_match = re.search(r"<Decide>(.*?)</Decide>", output, re.DOTALL)
        if decide_match:
            result["type"] = "decide"
            decide_content = decide_match.group(1)
            # Extract answer from within Decide block
            answer_match = re.search(r"<answer>\s*([ABCD])\s*</answer>", decide_content, re.IGNORECASE)
            if answer_match:
                result["answer"] = answer_match.group(1).upper()
            return result

        # Check for <Request> tag
        request_match = re.search(r"<Request>(.*?)</Request>", output, re.DOTALL)
        if request_match:
            result["type"] = "request"
            result["request_text"] = request_match.group(1).strip()
            return result

        # Fallback: try to find any answer tag (no explicit Decide/Request)
        answer_match = re.search(r"<answer>\s*([ABCD])\s*</answer>", output, re.IGNORECASE)
        if answer_match:
            result["type"] = "decide"
            result["answer"] = answer_match.group(1).upper()
            return result

        # Fallback: look for FINAL ANSWER pattern
        final_match = re.search(r"FINAL ANSWER:\s*([ABCD])", output, re.IGNORECASE)
        if final_match:
            result["type"] = "decide"
            result["answer"] = final_match.group(1).upper()
            return result

        # Fallback: look for standalone letter answer patterns like "Answer: A" or "the answer is B"
        standalone_match = re.search(r"(?:answer|choice|option)\s*(?:is|:)\s*([ABCD])\b", output, re.IGNORECASE)
        if standalone_match:
            result["type"] = "decide"
            result["answer"] = standalone_match.group(1).upper()
            return result

        # Last resort: find any isolated A/B/C/D near the end of output
        last_letter = re.findall(r"\b([ABCD])\b", output[-200:])
        if last_letter:
            result["type"] = "decide"
            result["answer"] = last_letter[-1].upper()
            return result

        # Default to decide with no answer (will trigger forced decision in the loop)
        result["type"] = "decide"
        return result

    def __call__(self, question_data, program_code, tools_definitions, analysis_data, prompt, retry_count=0):
        """Non-iterative analysis (backward-compatible single-pass mode)."""
        data_dict_str = self._format_data_dict(analysis_data)

        prompt_text = prompt.format(
            question=question_data["question"],
            program_code=program_code,
            tools_definitions=tools_definitions,
            data_dictionary=data_dict_str
        )

        if retry_count > 0:
            prompt_text += f"\n\nIMPORTANT: This is retry attempt {retry_count}. The previous analysis did not contain a clear final answer (A, B, C, or D). Please ensure your response includes a clear final answer in one of these formats:\n- <answer>A</answer>"

        output, messages = self.generator.generate(prompt_text)

        final_answer = self._extract_final_answer(output)

        if final_answer is None and retry_count < self.max_retries:
            print(f"No valid answer extracted on attempt {retry_count + 1}, retrying... (max retries: {self.max_retries})")
            return self.__call__(question_data, program_code, tools_definitions, analysis_data, prompt, retry_count + 1)

        analysis_result = {
            "question_id": question_data.get("id", "unknown"),
            "question": question_data["question"],
            "options": question_data.get("options", []),
            "program_code": program_code,
            "analysis_data": analysis_data,
            "full_analysis": output,
            "final_answer": final_answer,
            "prompt": prompt_text,
            "messages": messages,
            "model_name": self.generator.model_name,
            "retry_count": retry_count,
            "max_retries_reached": retry_count >= self.max_retries and final_answer is None
        }

        self.analyses.append(analysis_result)

        if final_answer is None:
            print(f"Warning: No valid answer extracted after {retry_count + 1} attempts (including initial attempt)")

        return final_answer, output

    def _format_data_dict(self, data_dict):
        """Convert the analysis data dictionary to a readable string format."""
        if not isinstance(data_dict, dict):
            return str(data_dict)

        formatted_str = ""
        for key, value in data_dict.items():
            formatted_str += f"{key}:\n"
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    formatted_str += f"  {sub_key}: {sub_value}\n"
            elif isinstance(value, list):
                formatted_str += f"  {value}\n"
            else:
                formatted_str += f"  {value}\n"
            formatted_str += "\n"

        return formatted_str

    def _extract_final_answer(self, output):
        """Extract the final answer (A, B, C, or D) from the analysis output."""
        # Look for <answer> tags first
        answer_match = re.search(r"<answer>\s*([ABCD])\s*</answer>", output, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        # Look for "FINAL ANSWER:" followed by a letter
        final_answer_match = re.search(r"FINAL ANSWER:\s*([ABCD])", output, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).upper()

        return None

    def save_all_analyses(self, results_folder_path):
        """Save all analyses to a JSON file."""
        analyses_path = os.path.join(results_folder_path, "analyses.json")

        with open(analyses_path, "w+") as file:
            json.dump(self.analyses, file, indent=2, default=str)

        return analyses_path
