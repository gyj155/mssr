import os
import re
import signal
import sys
import csv
from typing import Literal, List, Dict, Any, Union
import io
import base64

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from tqdm import tqdm
import json
import linecache
import runpy
import traceback
from engine.engine_utils import (
    Generator,
    get_methods_from_json,
    TimeoutException,
    timeout_handler,
    correct_indentation,
    replace_tabs_with_spaces,
)
from engine.predefined_modules import ModulesList


class Engine:
    def __init__(
        self,
        api_json=None,
        results_folder_path="",
        models_path="",
        dataset="mmsi-bench",
        vqa_model="gemini-3.1-flash-lite-preview",
        module_list=None
    ):
        self.api_json = api_json
        self.results_folder_path = results_folder_path
        print("Initializing modules")
        if module_list is None:
            self.modules_list = ModulesList(models_path=models_path, dataset=dataset, vqa_model=vqa_model)
        else:
            self.modules_list = module_list
        if api_json:
            self.api_methods, self.namespace = get_methods_from_json(self.api_json)
        else:
            self.api_methods = []
            self.namespace = {}

        self.namespace.update(self.modules_list.module_executes)
        self.trace_file_path = ""
        self.program_executable_path = ""
        self.result_file = ""
        self.execution_json = []
        self.output_csv_path = os.path.join(results_folder_path, "outputs.csv")

    def _preprocess_images_for_consistency(self, image_paths, images_folder_path):
        """
        Preprocess images using the same logic as VGGT's load_and_preprocess_images
        """
        processed_images = []
        target_size = 518
        mode = "crop"  # Using the same default mode as VGGT
        
        for image_path in image_paths:
            # Open image
            img = Image.open(os.path.join(images_folder_path, image_path))
            
            # Handle RGBA images
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            
            # Convert to RGB
            img = img.convert("RGB")
            
            width, height = img.size
            
            # Calculate new dimensions (same logic as load_and_preprocess_images)
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            
            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            
            # Center crop height if it's larger than target_size
            if new_height > target_size:
                # Crop from center
                top = (new_height - target_size) // 2
                img = img.crop((0, top, new_width, top + target_size))
            
            processed_images.append(img)
        
        return processed_images

    def execute_programs(
        self, programs, questions, images_folder_path
    ):

        folder_name = "program_execution"
        results_folder_path = os.path.join(
            self.results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path, exist_ok=True)

        for question, program_data in tqdm(
            zip(questions, programs), total=len(questions)
        ):
            exec_env_path = results_folder_path

            os.makedirs(exec_env_path, exist_ok=True)

            self.trace_file_path = os.path.join(exec_env_path, "trace.html")
            with open(self.trace_file_path, "w+") as f:
                f.write(f"<h1>Question: {question['question']}</h1>")
                for i, image_path in enumerate(question["image_paths"]):
                    image = Image.open(
                        os.path.join(images_folder_path, image_path)
                    )
                    image.thumbnail((400,400), Image.Resampling.LANCZOS)
                    rgb_image = image.convert("RGB")
                    image_io = io.BytesIO()
                    rgb_image.save(image_io, format="PNG")
                    image_bytes = base64.b64encode(image_io.getvalue()).decode("ascii")
                    
                    f.write(f"<h2>Image {i+1}</h2>\n")
                    f.write(
                        f"<img src='data:image/jpeg;base64,{image_bytes}'>\n"
                    )
            self.program_executable_path = os.path.join(
                exec_env_path, "executable_program.py"
            )
            self.result_file = os.path.join(exec_env_path, "result.json")

            if "image_paths" in question and question["image_paths"]:
                print('preprocessing images')
                images = self._preprocess_images_for_consistency(question["image_paths"], images_folder_path)
                image = images
            else:
                image = Image.open(
                    os.path.join(images_folder_path, question["image_filename"])
                )


            self.execution_json.append(
                self.run_program(
                    program_data, image, question, "execution", error_count=0
                )
            )

        execution_json_path = os.path.join(results_folder_path, "execution.json")

        with open(execution_json_path, "w+") as file:
            json.dump(self.execution_json, file)

        self.save_evaluation_accuracy(
            self.execution_json, results_folder_path, "execution"
        )

    def run_program(
        self,
        program_data,
        image,
        question,
        execution_type,
        scene_json=None,
        error_count=0,
    ):
        program = program_data["program"]

        # Reset call counters for each program run (including retries)
        self.modules_list.vqa_call_count = 0
        self.modules_list.find_obj_call_count = 0

        self.modules_list.set_trace_path(self.trace_file_path)
        execution_data = {}

        if isinstance(image, list):
            self.namespace.update(images=image)
        else:
            image = image.convert("RGB")
            self.namespace.update(image=image)


        try:
            if isinstance(program, list):
                program = program[0]
        except Exception as e:
            if error_count < 5:
                print("No program found")
                corrected_program_data = self.correct_program_error(
                    program_data, Exception("No program found"), question
                )
                return self.run_program(
                    corrected_program_data,
                    image,
                    question,
                    execution_type,
                    scene_json,
                    error_count + 1,
                )
            else:
                program = ""
        self._add_program_to_file(program)



        error = self._execute_file()
        if error and error_count < 5:
            print(error)
            corrected_program_data = self.correct_program_error(
                program_data, error, question
            )
        
            if os.path.exists(self.trace_file_path):
                os.remove(self.trace_file_path)
            
            return self.run_program(
                corrected_program_data,
                image,
                question,
                execution_type,
                scene_json,
                error_count + 1,
            )
        if error and error_count == 5:
            print(error)
            print("Error count is 5, aborting")
            return
            
        try:
            with open(self.result_file, "r") as f:
                result_namespace = json.load(f)
        except Exception as e:
            result_namespace = {"final_result": f"Error: {error}"}

        execution_data[execution_type] = {}
        execution_data[execution_type]["question"] = question
        execution_data[execution_type]["program"] = program
        execution_data[execution_type]["result_namespace"] = result_namespace
        if "final_result" in result_namespace:
            final_result = result_namespace["final_result"]
            if isinstance(final_result, bool):
                execution_data[execution_type]["answer"] = (
                    "yes" if final_result else "no"
                )
            elif isinstance(final_result, str):
                execution_data[execution_type]["answer"] = final_result.lower()
            else:
                execution_data[execution_type]["answer"] = final_result
        else:
            execution_data[execution_type]["answer"] = ""

        return execution_data

    def write_csv(self, filename, entries):
        # Check if entries have question_type field to determine field list
        has_question_type = any("question_type" in entry for entry in entries)
        
        if has_question_type:
            fields = [
                "question",
                "image_index",
                "question_type",
                "answer_type",
                "ground_truth",
                "prediction",
            ]
        else:
            fields = [
                "question",
                "image_index",
                "answer_type",
                "ground_truth",
                "prediction",
            ]
            
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def correct_program_error(self, program_data, error, question):
        messages = program_data["messages"]
        messages.append(
            {
                "role": "user",
                "content": f"\n There was an error in running the code: {error}. Try again and include the program between <program></program>",
            }
        )
        generator = Generator(program_data["model_name"])
        output, messages = generator.generate(None, messages)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
        program_data = {
            "image_paths": question["image_paths"],
            "question_index": question["id"],
            "program": program,
            "prompt": program_data["prompt"],
            "output": output,
            "messages": messages,
            "model_name": generator.model_name,
        }
        return program_data

    def write_summarized_results(self, csv_path, results_pth):
        mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        correct_at_threshold = {key: 0.0 for key in mra_thresholds}
        yn_correct = 0
        yn_n = 0
        num_ct_n = 0
        num_ct_correct = 0
        multi_correct = 0
        multi_n = 0
        num_other_n = 0

        reader = csv.reader(open(csv_path), delimiter=",")
        headers = next(reader, None)

        for row in reader:
            ans_type = row[-3]
            gt = row[-2]
            pred = row[-1]

            # Numeric (count)
            if ans_type == "int":
                num_ct_n += 1
                try:
                    pred = int(pred)
                except:
                    continue
                gt = int(gt)
                if gt == pred:
                    num_ct_correct += 1
            elif ans_type == "str":
                # Yes/No
                if gt.lower() in ["yes", "no"]:
                    yn_n += 1
                    try:
                        if gt.lower() == pred.lower():
                            yn_correct += 1
                    except:
                        continue
                # multi
                else:
                    multi_n += 1
                    try:
                        if gt.lower() == pred.lower():
                            multi_correct += 1
                    except:
                        continue
            elif ans_type == "float":
                # Numeric (other)
                num_other_n += 1
                for threshold in mra_thresholds:
                    try:
                        pred = float(pred)
                    except:
                        continue
                    gt = float(gt)
                    if abs(gt - pred) / gt < threshold:
                        correct_at_threshold[threshold] += 1.0

        # Compute AVG Accuracies
        yn_acc = yn_correct / yn_n if yn_n != 0 else None
        multi_acc = multi_correct / multi_n if multi_n != 0 else None
        num_ct_acc = num_ct_correct / num_ct_n if num_ct_n != 0 else None
        num_other_mra = 0

        if num_other_n != 0:
            for threshold in mra_thresholds:
                correct_at_threshold[threshold] = correct_at_threshold[threshold] / num_other_n
                num_other_mra += correct_at_threshold[threshold]

            num_other_mra = num_other_mra / len(mra_thresholds)
        else:
            num_other_mra = None

        with open(results_pth, "w") as f:
            f.write("-------- Summary Results --------\n")
            f.write(f"Yes/No Accuracy: {yn_acc}\n")
            f.write(f"Multiple Choice Accuracy: {multi_acc}\n")
            f.write(f"Numeric (count) Accuracy: {num_ct_acc}\n")
            f.write(f"Numeric (other) MRA: {num_other_mra}")

    def save_evaluation_accuracy(
        self,
        execution_data: list,
        results_path: str,
        execution_type: Literal["execution"],
    ):
        execution_results_path = os.path.join(results_path, execution_type + ".txt")
        execution_sheet_path = os.path.join(results_path, execution_type + ".csv")
        execution_results = ""
        execution_results += "\nResults:\n"
        num_correct = 0
        only_execution_correct = 0
        both_correct = 0

        json_results = []

        for execution in execution_data:
            question_data = execution[execution_type]["question"]
            question_index = question_data.get("question_index", question_data.get("id"))
            image_index = question_data.get("image_index", question_data.get("id"))
            
            answer_type = ""
            if "answer_type" in question_data:
                answer_type = str(question_data["answer_type"])
            elif "answer" in question_data:
                # Infer answer_type from the ground truth answer
                ans = str(question_data["answer"])
                if ans.lower() in ["yes", "no"]:
                    answer_type = "str"
                else:
                    try:
                        int(ans)
                        answer_type = "int"
                    except (ValueError, TypeError):
                        try:
                            float(ans)
                            answer_type = "float"
                        except (ValueError, TypeError):
                            answer_type = "str"
            
            json_results.append(
                {
                    "question": str(question_data["question"]),
                    "image_index": image_index,
                    "answer_type": answer_type,
                    "ground_truth": (
                        str(question_data["answer"])
                        if "answer" in question_data
                        else ""
                    ),
                    "prediction": str(execution[execution_type]["answer"]),
                }
            )

            execution_results += (
                f"Image: {image_index}\n"
            )
            execution_results += f"Question {question_index}: {question_data['question']}\n"
            if "program" in execution[execution_type]:
                execution_results += (
                    f"Program: {execution[execution_type]['program']}\n"
                )
            execution_results += (
                f"Predicted Answer: {execution[execution_type]['answer']}\n"
            )
            execution_results += (
                f"Correct Answer: {execution[execution_type]['question']['answer']}\n"
            )

            if isinstance(execution[execution_type]["answer"], bool):
                answer = "yes" if execution[execution_type]["answer"] else "no"
                if answer == execution[execution_type]["question"]["answer"]:
                    num_correct += 1
            elif (
                str(execution[execution_type]["answer"]).lower()
                == execution[execution_type]["question"]["answer"].lower()
            ):
                num_correct += 1

        execution_results += f"\nOnly Execution Correct: {float(only_execution_correct)/len(execution_data)}\n"
        execution_results += f"Accuracy: {float(num_correct)/len(execution_data)}\n"

        with open(execution_results_path, "w+") as file:
            file.write(execution_results)
            file.close()

        csv_path = os.path.join(results_path, f"{execution_type}.csv")
        self.write_csv(csv_path, json_results)

        # Write summarized results
        self.write_summarized_results(
            csv_path, os.path.join(results_path, "results.txt")
        )

    def _add_program_to_file(self, program):
        with open(self.program_executable_path, "w") as file:
            file.write("import math\n")
            file.write("import numpy as np\n")
            file.writelines(f"{method}\n" for method in self.api_methods)
            file.write("\n# PROGRAM STARTS HERE\n")

        new_program_content = [f"{line}\n" for line in program.split("\n")]

        write_namespace_code = f"""
print("Answer:", final_result)
# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {{k: v for k, v in globals().items() if is_serializable(v)}}

with open("{self.result_file}", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        """

        with open(self.program_executable_path, "a") as file:
            file.writelines(new_program_content)
            file.write(write_namespace_code)



    def _trace_execution(self, frame, event, arg):
        if event == "line":
            filename = frame.f_globals.get("__file__", None)
            if filename and os.path.basename(filename) == os.path.basename(
                self.program_executable_path
            ):
                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                if lineno > self.namespace_line:
                    return self._trace_execution
                if "import math" in line:
                    return self._trace_execution
                if "import json" in line:
                    self.namespace_line = lineno
                    return self._trace_execution
                # Get function name if we're inside one
                function_name = frame.f_code.co_name
                trace_line = f"<p>{lineno}: "
                if function_name and function_name != "<module>":
                    trace_line += f"[In method {function_name}] "
                trace_line += f"<code>{line}</code></p>\n"
                with open(self.trace_file_path, "a+") as f:
                    f.write(trace_line)

        return self._trace_execution

    def _execute_file(self):
        self.tracing_started = False
        self.namespace_line = sys.maxsize

        # Clear linecache to ensure the correct file content is read
        linecache.clearcache()
        
        # Keep a reference to the original trace function
        original_trace = sys.gettrace()
        if original_trace is None:
            original_trace = self._trace_execution

        def trace_wrapper(frame, event, arg):
            # Only trace lines from the main executable script
            if frame.f_code.co_filename == self.program_executable_path:
                sys.settrace(original_trace)
                return original_trace(frame, event, arg)
            # For any other file, do not trace, but keep the tracer active to regain control
            return trace_wrapper

        sys.settrace(trace_wrapper)

        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(200)
            exec_ns = runpy.run_path(self.program_executable_path, init_globals=self.namespace)
            signal.alarm(0)
            self._last_exec_namespace = exec_ns
        except TimeoutException as e:
            return e
        except Exception:
            return traceback.format_exc()
        finally:
            sys.settrace(None)
        return

    def _capture_snapshot(self):
        """Merge the last execution namespace into self.namespace for state preservation."""
        if not hasattr(self, '_last_exec_namespace') or self._last_exec_namespace is None:
            return
        EXCLUDE = {'__builtins__', '__name__', '__doc__', '__loader__',
                    '__spec__', '__package__', '__cached__', '__file__'}
        for k, v in self._last_exec_namespace.items():
            if k not in EXCLUDE and not k.startswith('__'):
                self.namespace[k] = v

    # ============= ITERATIVE TWO-AGENT PIPELINE =============

    def _save_question_results(self, combined_result, question_results_path):
        """Save combined results for a single question."""
        results_file = os.path.join(question_results_path, "combined_results.json")

        with open(results_file, "w") as f:
            json.dump(combined_result, f, indent=2, default=str)

        # Also save a simplified question result for easier viewing
        question_result_file = os.path.join(question_results_path, "question_result.json")
        simplified_result = {
            'question_id': combined_result['question_id'],
            'question': combined_result['question']['question'],
            'ground_truth': combined_result['ground_truth'],
            'predicted_answer': combined_result['final_answer'],
            'is_correct': combined_result['is_correct'],
            'question_type': combined_result['question_type'],
            'execution_successful': combined_result['success']
        }

        with open(question_result_file, "w") as f:
            json.dump(simplified_result, f, indent=2)

    def execute_iterative_pipeline(
        self, question, program_agent, analysis_agent, images_folder_path,
        question_results_path, max_iterations=5,
    ):
        """Execute the iterative two-agent pipeline for a single question.

        The Perception Agent (PA) collects data, then the Reasoning Agent (RA)
        curates the information set and either decides on an answer or requests
        more data. This loop repeats until RA decides or max iterations are hit.

        Args:
            question: Question data dict
            program_agent: ProgramAgent instance
            analysis_agent: AnalysisAgent instance
            images_folder_path: Path to images
            question_results_path: Path to save results for this question
            max_iterations: Maximum number of PA-RA iterations

        Returns:
            dict with combined results
        """
        from prompts.modules import MODULES_SIGNATURES_MMSI
        from prompts.data_collection_prompt import (
            DATA_COLLECTION_PROMPT_INITIAL,
            DATA_COLLECTION_PROMPT_TARGETED,
        )

        os.makedirs(question_results_path, exist_ok=True)
        tools_definitions = MODULES_SIGNATURES_MMSI

        # Preprocess images once
        images = self._preprocess_images_for_consistency(
            question["image_paths"], images_folder_path
        )
        self.namespace.update({"images": images})

        analysis_data = {}
        all_program_codes = []
        iteration_log = []
        final_answer = None

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            iter_results_path = os.path.join(question_results_path, f"iteration_{iteration}")
            os.makedirs(iter_results_path, exist_ok=True)

            # === Perception Agent (PA) phase ===
            if iteration == 0:
                # First iteration: broad data collection
                print(f"[PA] Generating initial broad data collection program...")
                program, output = program_agent(
                    question, DATA_COLLECTION_PROMPT_INITIAL,
                    images_folder_path=images_folder_path
                )
            else:
                # Subsequent iterations: targeted collection based on RA request
                request_text = iteration_log[-1]["request_text"]
                info_set_text = analysis_agent._format_data_dict(analysis_data)
                print(f"[PA] Generating targeted program for request: {request_text[:100]}...")
                program, output = program_agent.generate_targeted(
                    question, DATA_COLLECTION_PROMPT_TARGETED,
                    request_text, info_set_text,
                    images_folder_path=images_folder_path
                )

            # Execute the PA's program
            program_code = program[0] if program else ""
            all_program_codes.append(program_code)

            if not program_code:
                print(f"[PA] No program generated at iteration {iteration}")
                iteration_log.append({
                    "iteration": iteration, "pa_error": "No program generated",
                    "type": "error",
                })
                continue

            # Build and execute program
            exec_result = self._execute_pa_program(
                program_code, question, images_folder_path,
                iter_results_path, iteration,
                program_data=program_agent.programs[-1],
            )

            if not exec_result["execution_successful"]:
                print(f"[PA] Execution failed: {exec_result.get('error_info', 'unknown')}")
                iteration_log.append({
                    "iteration": iteration, "pa_error": exec_result.get("error_info"),
                    "type": "error",
                })
                continue

            # Capture namespace snapshot for next iteration
            self._capture_snapshot()

            # Merge new data into the cumulative analysis_data
            new_data = exec_result.get("analysis_data", {})
            analysis_data.update(new_data)
            print(f"[PA] Collected {len(new_data)} new items. Total items: {len(analysis_data)}")

            # === Reasoning Agent (RA) phase ===
            info_set_text = analysis_agent._format_data_dict(analysis_data)
            combined_code = "\n\n# ---\n\n".join(all_program_codes)

            is_last_iteration = (iteration == max_iterations - 1)
            print(f"[RA] Curating information set ({len(analysis_data)} items)...{' [LAST ITER]' if is_last_iteration else ''}")
            ra_result = analysis_agent.curate_and_decide(
                question, combined_code, tools_definitions, info_set_text,
                iteration=iteration, force_decide=is_last_iteration,
            )

            iter_entry = {
                "iteration": iteration,
                "type": ra_result["type"],
                "answer": ra_result.get("answer"),
                "request_text": ra_result.get("request_text"),
                "num_items_before": len(analysis_data),
                "curated_set": ra_result.get("curated_set", ""),
            }
            iteration_log.append(iter_entry)

            # Save iteration results
            self._save_iteration_results(iter_results_path, iteration, exec_result, ra_result, analysis_data)

            if ra_result["type"] == "decide":
                final_answer = ra_result.get("answer")
                if final_answer:
                    print(f"[RA] Decision made at iteration {iteration + 1}: {final_answer}")
                else:
                    print(f"[RA] Decided but no valid answer extracted at iteration {iteration + 1}")
                break
            else:
                print(f"[RA] Requesting more data: {ra_result['request_text'][:100]}...")

        # If max iterations reached without decision, force a decision
        if final_answer is None and iteration_log:
            print(f"[MSSR] Max iterations reached. Forcing final decision...")
            info_set_text = analysis_agent._format_data_dict(analysis_data)
            combined_code = "\n\n# ---\n\n".join(all_program_codes)
            forced_result = analysis_agent.curate_and_decide(
                question, combined_code, tools_definitions, info_set_text,
                iteration=max_iterations, force_decide=True,
            )
            final_answer = forced_result.get("answer")
            iteration_log.append({
                "iteration": max_iterations,
                "type": "forced_decide",
                "answer": final_answer,
            })

        # Calculate correctness
        is_correct = False
        if final_answer and 'answer' in question:
            is_correct = str(question['answer']).lower() == str(final_answer).lower()

        combined_result = {
            "question_id": question["id"],
            "question": question,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "ground_truth": question.get('answer', ''),
            "question_type": question.get('question_type', 'Unknown'),
            "success": final_answer is not None,
            "iteration_log": iteration_log,
            "all_program_codes": all_program_codes,
            "final_analysis_data": analysis_data,
            "data_collection_result": {
                "analysis_data": analysis_data,
                "execution_successful": True,
                "error_info": None,
                "program_code": "\n\n".join(all_program_codes),
            },
        }

        # Save combined results
        self._save_question_results(combined_result, question_results_path)
        return combined_result

    def _execute_pa_program(
        self, program_code, question, images_folder_path,
        results_path, iteration, program_data=None, error_count=0,
    ):
        """Execute a perception agent's program with error recovery.

        For iteration 0, includes preprocessing (VGGT, ground plane detection).
        For subsequent iterations, skips preprocessing (uses snapshot).
        """
        self.modules_list.vqa_call_count = 0
        self.modules_list.find_obj_call_count = 0

        self.trace_file_path = os.path.join(results_path, "trace.html")
        self.program_executable_path = os.path.join(results_path, "executable_program.py")
        self.result_file = os.path.join(results_path, "result.json")

        # Initialize trace file
        with open(self.trace_file_path, "w+") as f:
            f.write(f"<h1>Iteration {iteration} - Data Collection</h1>")
            f.write(f"<h2>Question: {question['question']}</h2>")
            for i, image_path in enumerate(question["image_paths"]):
                image = Image.open(os.path.join(images_folder_path, image_path))
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                rgb_image = image.convert("RGB")
                image_io = io.BytesIO()
                rgb_image.save(image_io, format="PNG")
                image_bytes = base64.b64encode(image_io.getvalue()).decode("ascii")
                f.write(f"<h3>Image {i+1}</h3>\n")
                f.write(f"<img src='data:image/jpeg;base64,{image_bytes}'>\n")

        self.modules_list.set_trace_path(self.trace_file_path)

        # Build the executable file
        with open(self.program_executable_path, "w+") as file:
            file.write("import math\n")
            file.write("import numpy as np\n")

            if iteration == 0:
                # First iteration: include preprocessing
                file.write("""
# PREPROCESSING CODE STARTS HERE
extrinsics, intrinsics, depth_maps, world_points = get_geo_info(images)
ground_info = ground_plane_detection(images)
ground_normal = ground_info["ground_normal"]
ground_centroid = ground_info["ground_centroid"]

""")

            file.write(f"\n# PROGRAM (iteration {iteration})\n")
            file.write(program_code + "\n\n")

            # Save analysis_data to JSON
            file.write(f"""
# SAVE ANALYSIS DATA
import json

def is_serializable(obj):
    try:
        json.dumps(obj, default=str)
    except (TypeError, OverflowError):
        return False
    return True

analysis_data_to_save = globals().get('analysis_data', {{}})
if isinstance(analysis_data_to_save, dict):
    serializable_data = {{k: v for k, v in analysis_data_to_save.items() if is_serializable(v)}}
else:
    serializable_data = {{"error": "analysis_data is not a dictionary", "raw": str(analysis_data_to_save)}}

result_to_save = {{
    "analysis_data": serializable_data,
    "execution_successful": True
}}

with open("{self.result_file}", "w+") as result_file_handle:
    json.dump(result_to_save, result_file_handle, indent=2, default=str)
""")

        # Execute
        error = self._execute_file()

        if error and error_count < 5:
            print(f"Execution error (attempt {error_count + 1}): {error}")
            if program_data:
                corrected = self.correct_program_error(program_data, error, question)
                new_code = corrected["program"]
                if isinstance(new_code, list):
                    new_code = new_code[0] if new_code else ""
                if os.path.exists(self.trace_file_path):
                    os.remove(self.trace_file_path)
                return self._execute_pa_program(
                    new_code, question, images_folder_path,
                    results_path, iteration, corrected, error_count + 1,
                )

            return {
                "analysis_data": {},
                "execution_successful": False,
                "error_info": str(error),
            }

        if error and error_count >= 5:
            return {
                "analysis_data": {},
                "execution_successful": False,
                "error_info": f"Max retries reached: {error}",
            }

        # Load results
        try:
            assert os.path.exists(self.result_file), "Result file was not created"
            with open(self.result_file, "r") as f:
                saved_result = json.load(f)
            return {
                "analysis_data": saved_result.get("analysis_data", {}),
                "execution_successful": saved_result.get("execution_successful", True),
                "error_info": None,
            }
        except Exception as e:
            return {
                "analysis_data": {},
                "execution_successful": False,
                "error_info": str(e),
            }

    def _save_iteration_results(self, iter_path, iteration, exec_result, ra_result, analysis_data):
        """Save per-iteration results."""
        result = {
            "iteration": iteration,
            "execution_successful": exec_result.get("execution_successful", False),
            "analysis_data_snapshot": exec_result.get("analysis_data", {}),
            "ra_decision_type": ra_result["type"],
            "ra_answer": ra_result.get("answer"),
            "ra_request": ra_result.get("request_text"),
            "ra_full_output": ra_result.get("full_output", ""),
            "cumulative_analysis_data": analysis_data,
        }
        with open(os.path.join(iter_path, "iteration_result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
