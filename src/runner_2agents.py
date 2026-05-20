import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
import json
from engine.predefined_modules import ModulesList
import traceback

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agents.agents import ProgramAgent, AnalysisAgent
from engine.engine import Engine
from prompts.modules import MODULES_SIGNATURES_MMSI


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class MSSRService:
    """Interactive MSSR (Minimal Sufficient Spatial Reasoner) service.

    Implements the iterative dual-agent pipeline:
    Perception Agent (PA) collects data -> Reasoning Agent (RA) curates and decides/requests -> loop.
    """

    def __init__(self, args):
        print("Initializing MSSR Service...")
        print("This may take a while for the first time...")

        self.args = args

        print("Loading questions data...")
        with open(args.annotations_json, "r") as file:
            self.questions_data = json.load(file)

        print("Loading predefined modules...")
        self.modules_list = ModulesList(
            models_path=args.models_path,
            dataset=args.dataset,
            vqa_model=args.model_name
        )

        print("MSSR Service initialized successfully!")
        print("All models loaded and ready.")

    def process_question(self, question_id):
        """Process a single question using the iterative two-agent pipeline."""
        print(f"\nProcessing question {question_id} with MSSR iterative pipeline...")

        questions = [q for q in self.questions_data if q['id'] == question_id]
        if len(questions) == 0:
            print(f"Question ID {question_id} not found")
            return False

        assert len(questions) == 1, f"Multiple questions found with ID {question_id}"
        question = questions[0]
        question['question'] = question['question'] + f'  Total number of images: {len(question["image_paths"])}'

        results_folder_path = os.path.join(
            self.args.results_pth,
            f"mssr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_question_{question_id}"
        )
        os.makedirs(results_folder_path)

        # Create agents
        program_agent = ProgramAgent(
            model_name=self.args.model_name,
            dataset=self.args.dataset,
            predef_signatures=MODULES_SIGNATURES_MMSI,
            prompt_mode=self.args.prompt_mode
        )

        analysis_agent = AnalysisAgent(
            model_name=self.args.model_name,
            dataset=self.args.dataset,
            write_results=True,
            max_retries=3
        )

        # Create engine
        engine = Engine(
            results_folder_path=results_folder_path,
            module_list=self.modules_list,
            dataset=self.args.dataset,
            vqa_model=self.args.model_name,
            models_path=self.args.models_path
        )

        # Execute iterative pipeline
        result = engine.execute_iterative_pipeline(
            question=question,
            program_agent=program_agent,
            analysis_agent=analysis_agent,
            images_folder_path=self.args.image_pth,
            question_results_path=results_folder_path,
            max_iterations=self.args.max_iterations,
        )

        # Save analysis log
        analysis_agent.save_all_analyses(results_folder_path)

        # Print summary
        if result:
            print(f"\nQuestion {question_id} completed!")
            print(f"  Final answer: {result['final_answer']}")
            print(f"  Correct answer: {question.get('answer', 'N/A')}")
            print(f"  Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
        else:
            print(f"No result returned for question {question_id}")

        print(f"Results saved to: {results_folder_path}")
        return True

    def run_interactive(self):
        """Run the interactive service."""
        print("\n" + "="*60)
        print("MSSR Service is now ready!")
        print("Commands:")
        print("  - Enter question ID: 123")
        print("  - Enter 'quit' to exit")
        print("="*60)

        while True:
            user_input = input("\nEnter question ID (or 'quit'): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            try:
                question_id = int(user_input)
                success = self.process_question(question_id)
                if success:
                    print(f"Question {question_id} completed successfully")
                else:
                    print(f"Failed to process question {question_id}")

            except ValueError:
                print("Please enter a valid question ID or 'quit'")
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(traceback.format_exc())
                continue


def main():
    set_seeds(42)
    parser = argparse.ArgumentParser(description="MSSR - Minimal Sufficient Spatial Reasoner")
    parser.add_argument("--dataset", default="mmsi-bench", choices=["mmsi-bench"],
                       help="Name of dataset (default: %(default)s)")
    parser.add_argument("--annotations-json", default="dataset/MMSI-Bench/mmsi_bench.json",
                       help="Path to annotations JSON file")
    parser.add_argument("--image-pth", default="dataset/MMSI-Bench",
                       help="Path to directory containing images")
    parser.add_argument("--models-path", default="src/models",
                       help="Path to directory containing models")
    parser.add_argument("--results-pth", default="src/results/",
                       help="Path to directory to save results")
    parser.add_argument("--model-name", default="gemini-3.1-flash-lite-preview",
                       choices=[
                           "gemini-3.1-flash-lite-preview",
                           "gemini-2.5-flash",
                           "gemini-2.5-pro",
                           "gemini-2.0-flash",
                           "gpt-4o",
                           "gpt-4o-mini",
                       ],
                       help="Model name (default: %(default)s)")
    parser.add_argument("--prompt-mode", default="text", choices=["text", "with-image"],
                       help="Prompt mode (default: %(default)s)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum PA-RA iterations (default: %(default)s)")

    args = parser.parse_args()

    service = MSSRService(args)
    service.run_interactive()


if __name__ == "__main__":
    main()

"""
Usage examples:

python src/runner_2agents.py --model-name gemini-3.1-flash-lite-preview --prompt-mode text

python src/runner_2agents.py --model-name gemini-2.5-flash --prompt-mode with-image

python src/runner_2agents.py --model-name gemini-2.5-flash --prompt-mode text \\
    --annotations-json dataset/ViewSpatial-Bench/ViewSpatial-Bench_processed.json \\
    --image-pth dataset/ViewSpatial-Bench
"""
