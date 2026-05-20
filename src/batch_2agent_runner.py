import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import traceback

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agents.agents import ProgramAgent, AnalysisAgent
from engine.engine import Engine
from engine.predefined_modules import ModulesList
from prompts.modules import MODULES_SIGNATURES_MMSI


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def calculate_current_accuracy(all_results, results_folder_path):
    """Calculate and save current accuracy after each question, including by question type."""
    if not all_results:
        return 0.0, {}

    num_correct = 0
    total_questions = len(all_results)
    question_type_stats = {}

    for result in all_results:
        if result and 'final_answer' in result and 'question' in result:
            question_data = result['question']
            predicted_answer = result['final_answer']
            question_type = question_data.get('question_type', 'Unknown')

            if question_type not in question_type_stats:
                question_type_stats[question_type] = {"correct": 0, "total": 0}
            question_type_stats[question_type]["total"] += 1

            if 'answer' in question_data and predicted_answer:
                ground_truth = str(question_data['answer']).lower()
                prediction = str(predicted_answer).lower()
                if ground_truth == prediction:
                    num_correct += 1
                    question_type_stats[question_type]["correct"] += 1

    accuracy = num_correct / total_questions if total_questions > 0 else 0.0

    accuracy_file = os.path.join(results_folder_path, "current_accuracy.txt")
    with open(accuracy_file, "w") as f:
        f.write(f"Current Overall Accuracy: {accuracy:.4f} ({num_correct}/{total_questions})\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== Accuracy by Question Type ===\n")
        for q_type, stats in sorted(question_type_stats.items()):
            type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            f.write(f"{q_type}: {type_accuracy:.4f} ({stats['correct']}/{stats['total']})\n")

    return accuracy, question_type_stats


def main():
    set_seeds(42)
    parser = argparse.ArgumentParser(description="MSSR Batch Runner - Iterative Two-Agent Pipeline")
    parser.add_argument("--dataset", default="mmsi-bench", choices=["mmsi-bench"],
                       help="Name of dataset (default: %(default)s)")
    parser.add_argument("--annotations-json", default="dataset/MMSI-Bench/mmsi_bench.json",
                       help="Path to json file with questions")
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
                       help="Maximum PA-RA iterations per question (default: %(default)s)")
    parser.add_argument("--subset-ratio", type=float, default=1.0,
                       help="Fraction of dataset to evaluate (default: 1.0 = full)")

    args = parser.parse_args()

    print("Starting MSSR Batch Run (Iterative Pipeline)...")

    # Load questions
    print(f"Loading questions from: {args.annotations_json}")
    with open(args.annotations_json, "r") as file:
        questions_data = json.load(file)

    # Subsample if requested
    if args.subset_ratio < 1.0:
        n_subset = max(1, int(len(questions_data) * args.subset_ratio))
        random.shuffle(questions_data)
        questions_data = questions_data[:n_subset]
        print(f"Using {n_subset}/{len(questions_data)} questions ({args.subset_ratio*100:.0f}% subset)")

    # Add image count to questions
    for q in questions_data:
        if "image_paths" in q and len(q["image_paths"]) > 1:
            q["question"] = q["question"] + f'\nTotal number of images: {len(q["image_paths"])}'

    # Create results folder
    results_folder_path = os.path.join(
        args.results_pth,
        f"mssr_batch_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.annotations_json.split('/')[-1].split('.')[0]}"
    )
    os.makedirs(results_folder_path)
    print(f"Results will be saved to: {results_folder_path}")

    # Initialize heavy models once
    print("Initializing modules (this may take a while)...")
    modules_list = ModulesList(
        models_path=args.models_path, dataset=args.dataset, vqa_model=args.model_name
    )
    print("Modules initialized.")

    all_results = []

    print("\n" + "=" * 20 + " Processing questions with MSSR iterative pipeline " + "=" * 20)

    for i, question in enumerate(tqdm(questions_data, desc="MSSR Batch")):
        question_id = question['id']
        print(f"\n[{i+1}/{len(questions_data)}] Question {question_id}...")

        question_results_path = os.path.join(results_folder_path, f"question_{question_id}")
        os.makedirs(question_results_path, exist_ok=True)

        # Create fresh agents for each question
        program_agent = ProgramAgent(
            model_name=args.model_name,
            dataset=args.dataset,
            predef_signatures=MODULES_SIGNATURES_MMSI,
            prompt_mode=args.prompt_mode
        )

        analysis_agent = AnalysisAgent(
            model_name=args.model_name,
            dataset=args.dataset,
            write_results=True
        )

        # Create engine (reuses heavy modules)
        engine = Engine(
            results_folder_path=question_results_path,
            module_list=modules_list,
            dataset=args.dataset,
            vqa_model=args.model_name,
            models_path=args.models_path
        )

        try:
            result = engine.execute_iterative_pipeline(
                question=question,
                program_agent=program_agent,
                analysis_agent=analysis_agent,
                images_folder_path=args.image_pth,
                question_results_path=question_results_path,
                max_iterations=args.max_iterations,
            )

            analysis_agent.save_all_analyses(question_results_path)
            result['question'] = question
            all_results.append(result)

            print(f"  Answer: {result['final_answer']} | GT: {question.get('answer', 'N/A')} | "
                  f"{'CORRECT' if result['is_correct'] else 'INCORRECT'}")

        except Exception as e:
            print(f"  Error: {e}")
            print(traceback.format_exc())
            all_results.append({
                "question_id": question_id,
                "question": question,
                "final_answer": None,
                "is_correct": False,
                "success": False,
                "error": str(e),
                "data_collection_result": {"error_info": str(e)},
            })

        # Update running accuracy
        current_accuracy, _ = calculate_current_accuracy(all_results, results_folder_path)
        answered = len([r for r in all_results if r and r.get('final_answer')])
        print(f"  Running accuracy: {current_accuracy:.4f} ({answered}/{len(all_results)})")

    # Final evaluation
    print("\n" + "=" * 20 + " Final MSSR Evaluation " + "=" * 20)

    final_results_path = os.path.join(results_folder_path, "final_results.json")
    with open(final_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    successful = [r for r in all_results if r and r.get('final_answer')]
    correct_count = sum(1 for r in successful if r.get('is_correct', False))
    total = len(questions_data)

    question_type_stats = {}
    for r in successful:
        q_type = r.get('question_type', r.get('question', {}).get('question_type', 'Unknown'))
        if q_type not in question_type_stats:
            question_type_stats[q_type] = {"correct": 0, "total": 0}
        question_type_stats[q_type]["total"] += 1
        if r.get('is_correct'):
            question_type_stats[q_type]["correct"] += 1

    summary_path = os.path.join(results_folder_path, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("MSSR Batch Processing Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Questions: {total}\n")
        f.write(f"Successfully Answered: {len(successful)}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Overall Accuracy: {correct_count / total:.4f}\n")
        f.write(f"Model: {args.model_name}\n\n")
        f.write("=== Accuracy by Question Type ===\n")
        for q_type, stats in sorted(question_type_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            f.write(f"{q_type}: {acc:.4f} ({stats['correct']}/{stats['total']})\n")

    print(f"Overall Accuracy: {correct_count / total:.4f} ({correct_count}/{total})")
    print(f"\nQuestion Type Breakdown:")
    for q_type, stats in sorted(question_type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {q_type}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    print(f"\nResults saved to: {results_folder_path}")


if __name__ == "__main__":
    main()

"""
Usage examples:

# Full run with gemini-2.5-flash
CUDA_VISIBLE_DEVICES=0 python src/batch_2agent_runner.py \\
    --model-name gemini-2.5-flash \\
    --annotations-json dataset/MMSI-Bench/mmsi_bench.json

# ViewSpatial-Bench evaluation
CUDA_VISIBLE_DEVICES=0 python src/batch_2agent_runner.py \\
    --model-name gemini-2.5-flash \\
    --annotations-json dataset/ViewSpatial-Bench/ViewSpatial-Bench_processed.json \\
    --image-pth dataset/ViewSpatial-Bench
"""
