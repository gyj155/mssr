import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import time
import multiprocessing as mp
from multiprocessing import Manager, Lock, Process, Queue
import threading
from typing import List, Dict, Any, Tuple
import logging

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agents.agents import ProgramAgent, AnalysisAgent
from engine.engine import Engine
from engine.predefined_modules import ModulesList
from prompts.modules import MODULES_SIGNATURES_MMSI


def format_time(seconds):
    """Format seconds into hours:minutes:seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_logging(gpu_id: int, results_folder: str):
    """Setup logging for each GPU process - redirect all output to log file"""
    log_file = os.path.join(results_folder, f"gpu_{gpu_id}.log")
    
    logger = logging.getLogger(f"GPU-{gpu_id}")
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f'[GPU-{gpu_id}] %(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    import sys
    sys.stdout = open(log_file, 'a')
    sys.stderr = sys.stdout
    
    return logger


class SharedStatistics:
    """Manages shared statistics across all processes"""
    
    def __init__(self, manager: Manager):
        # Shared data structures
        self.results_list = manager.list()  # All results
        self.processed_count = manager.Value('i', 0)  # Total processed questions
        self.correct_count = manager.Value('i', 0)  # Total correct answers
        self.question_type_stats = manager.dict()  # Stats by question type
        self.start_time = manager.Value('d', 0.0)  # Start time for progress calculation
        self.lock = manager.Lock()  # Synchronization lock
    
    def set_start_time(self, start_time: float):
        """Set the start time for progress calculation"""
        with self.lock:
            self.start_time.value = start_time
        
    def update_stats(self, result: Dict[str, Any], question: Dict[str, Any]):
        """Thread-safe update of shared statistics"""
        with self.lock:
            self.results_list.append(result)
            self.processed_count.value += 1
            
            question_type = question.get('question_type', 'Unknown')
            
            if question_type not in self.question_type_stats:
                stats = {'correct': 0, 'total': 0}
            else:
                stats = self.question_type_stats[question_type]
            
            stats['total'] += 1
            if result and result.get('final_answer') is not None:
                if 'answer' in question and result.get('final_answer'):
                    ground_truth = str(question['answer']).lower()
                    prediction = str(result['final_answer']).lower()
                    
                    if ground_truth == prediction:
                        self.correct_count.value += 1
                        stats['correct'] += 1
            
            self.question_type_stats[question_type] = stats
    
    def get_current_stats(self) -> Tuple[float, Dict[str, Dict[str, int]]]:
        """Get current accuracy and question type statistics"""
        with self.lock:
            total = self.processed_count.value
            correct = self.correct_count.value
            accuracy = correct / total if total > 0 else 0.0
            type_stats = {}
            for q_type, stats in self.question_type_stats.items():
                type_stats[q_type] = dict(stats)
            
            return accuracy, type_stats
    
    def save_current_stats(self, results_folder: str):
        """Save current statistics to file"""
        accuracy, type_stats = self.get_current_stats()
        
        stats_file = os.path.join(results_folder, "current_accuracy.txt")
        with open(stats_file, "w") as f:
            f.write(f"Current Overall Accuracy: {accuracy:.4f} ({self.correct_count.value}/{self.processed_count.value})\n")
            f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== Accuracy by Question Type ===\n")
            for q_type, stats in sorted(type_stats.items()):
                type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                f.write(f"{q_type}: {type_accuracy:.4f} ({stats['correct']}/{stats['total']})\n")


def gpu_worker_process(
    gpu_id: int,
    questions_subset: List[Dict[str, Any]],
    args: argparse.Namespace,
    shared_stats: SharedStatistics,
    results_folder: str,
    progress_queue: Queue
):
    """Worker process that runs on a specific GPU"""
    
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    logger_prefix = f"GPU-{gpu_id}"
    
    logger = setup_logging(gpu_id, results_folder)
    logger.info(f"Starting {logger_prefix} worker with {len(questions_subset)} questions")
    logger.info(f"Device: {device}")

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    logger.info(f"Current CUDA device: {current_device} ({device_name})")
    
    gpu_results_folder = os.path.join(results_folder, f"gpu_{gpu_id}")
    os.makedirs(gpu_results_folder, exist_ok=True)
    
    try:
        set_seeds(42 + gpu_id)
        logger.info("Initializing modules list...")
        modules_list = ModulesList(
            models_path=args.models_path,
            dataset=args.dataset,
            vqa_model=args.model_name
        )
        logger.info(f"Moving models to {device}...")

        modules_list.grounding_dino = modules_list.grounding_dino.to(device)
        modules_list.vggt_model = modules_list.vggt_model.to(device)
        modules_list.sam2_predictor.model = modules_list.sam2_predictor.model.to(device)
        modules_list.device = str(device)
        
        # Update device in individual modules that have device attribute
        for module in modules_list.modules:
            if hasattr(module, 'device'):
                module.device = str(device)
            if hasattr(module, 'grounding_dino') and module.grounding_dino is not None:
                module.grounding_dino = module.grounding_dino.to(device)
            if hasattr(module, 'vggt_model') and module.vggt_model is not None:
                module.vggt_model = module.vggt_model.to(device)
            if hasattr(module, 'sam2_predictor') and module.sam2_predictor is not None:
                if hasattr(module.sam2_predictor, 'model'):
                    module.sam2_predictor.model = module.sam2_predictor.model.to(device)
        
        model_device = next(modules_list.grounding_dino.parameters()).device
        logger.info(f"GroundingDINO parameters are on device: {model_device}")

        for i, question in enumerate(questions_subset):
            question_id = question['id']
            logger.info(f"[{i+1}/{len(questions_subset)}] Processing question {question_id}")
            
            question_results_path = os.path.join(gpu_results_folder, f"question_{question_id}")
            os.makedirs(question_results_path, exist_ok=True)
            
            current_question_list = [question]
            result = None
            
            try:
                # Create agents
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

                # Execute iterative pipeline with GPU-specific modules
                engine = Engine(
                    results_folder_path=question_results_path,
                    module_list=modules_list,
                    dataset=args.dataset,
                    vqa_model=args.model_name,
                    models_path=args.models_path
                )

                result = engine.execute_iterative_pipeline(
                    question=question,
                    program_agent=program_agent,
                    analysis_agent=analysis_agent,
                    images_folder_path=args.image_pth,
                    question_results_path=question_results_path,
                    max_iterations=getattr(args, 'max_iterations', 5),
                )

                analysis_agent.save_all_analyses(question_results_path)

                if result:
                    result['question'] = question

                    is_correct = False
                    if result.get('final_answer') and 'answer' in question:
                        ground_truth = str(question['answer']).lower()
                        prediction = str(result['final_answer']).lower()
                        is_correct = ground_truth == prediction
                    
                    result['is_correct'] = is_correct
                    
                    # Save individual question result
                    question_result_file = os.path.join(question_results_path, "question_result.json")
                    with open(question_result_file, "w") as f:
                        json.dump({
                            'gpu_id': gpu_id,
                            'question_id': question_id,
                            'question': question['question'],
                            'ground_truth': question.get('answer', ''),
                            'predicted_answer': result.get('final_answer', ''),
                            'is_correct': is_correct,
                            'question_type': question.get('question_type', 'Unknown'),
                            'execution_successful': result.get('success', False)
                        }, f, indent=2)
                    
                    logger.info(f"Question {question_id} completed - {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    
                else:
                    # Handle failed questions
                    logger.warning(f"No results for question {question_id}")
                    question_result_file = os.path.join(question_results_path, "question_result.json")
                    with open(question_result_file, "w") as f:
                        json.dump({
                            'gpu_id': gpu_id,
                            'question_id': question_id,
                            'question': question['question'],
                            'ground_truth': question.get('answer', ''),
                            'predicted_answer': None,
                            'is_correct': False,
                            'question_type': question.get('question_type', 'Unknown'),
                            'execution_successful': False
                        }, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing question {question_id}: {str(e)}")
                result = None
            
            # Update shared statistics
            shared_stats.update_stats(result, question)
            
            # Save current stats after each question
            shared_stats.save_current_stats(results_folder)
            
            # Report progress
            progress_queue.put({
                'gpu_id': gpu_id,
                'question_id': question_id,
                'completed': i + 1,
                'total': len(questions_subset)
            })
            
        logger.info(f"GPU {gpu_id} completed all questions")
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} encountered fatal error: {str(e)}")
        raise


def progress_monitor(progress_queue: Queue, total_questions: int, num_gpus: int, shared_stats: 'SharedStatistics'):
    """Monitor and display progress from all GPU processes with time estimation"""
    completed_per_gpu = {i: 0 for i in range(num_gpus)}
    
    while True:
        try:
            progress_info = progress_queue.get(timeout=1)
            if progress_info is None:
                break
                
            gpu_id = progress_info['gpu_id']
            completed_per_gpu[gpu_id] = progress_info['completed']
            
            total_completed = sum(completed_per_gpu.values())
            progress_pct = (total_completed / total_questions) * 100
            current_time = time.time()
            with shared_stats.lock:
                start_time = shared_stats.start_time.value
            
            if start_time > 0 and total_completed > 0:
                elapsed_time = current_time - start_time
                avg_time_per_question = elapsed_time / total_completed
                estimated_total_time = avg_time_per_question * total_questions
                remaining_time = estimated_total_time - elapsed_time
                remaining_time = max(0, remaining_time)
                
                time_info = f"Elapsed: {format_time(elapsed_time)} | Est. Total: {format_time(estimated_total_time)} | Remaining: {format_time(remaining_time)}"
            else:
                time_info = "Calculating time..."
            gpu_info = " | ".join([f"GPU{i}: {completed_per_gpu[i]}" for i in range(num_gpus)])
            print(f"\rProgress: {total_completed}/{total_questions} ({progress_pct:.1f}%) - {gpu_info} - {time_info}",
                  end="", flush=True)
                  
        except Exception as e:
            continue


def main():
    set_seeds(42)
    parser = argparse.ArgumentParser(description="Distributed Two-Agent MSSR Batch Runner")
    
    # Original arguments
    parser.add_argument("--dataset", default="mmsi-bench", choices=["mmsi-bench"])
    parser.add_argument("--annotations-json", default="dataset/MMSI-Bench/mmsi_bench.json")
    parser.add_argument("--image-pth", default="dataset/MMSI-Bench")
    parser.add_argument("--models-path", default="src/models")
    parser.add_argument("--results-pth", default="src/results/")
    parser.add_argument("--model-name", default="gemini-3.1-flash-lite-preview",
                       choices=["gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gemini-2.5-pro",
                               "gemini-2.0-flash", "gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--prompt-mode", default="text", choices=["text", "with-image"])
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum PA-RA iterations per question (default: 5)")

    # Distributed arguments
    parser.add_argument("--gpu-ids", type=str, required=True,
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    
    args = parser.parse_args()
    
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    
    assert torch.cuda.is_available(), "CUDA is not available"
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
        
    print("GPU Information:")
    for gpu_id in gpu_ids:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"  GPU {gpu_id}: {gpu_name}")
    
    print(f"Starting Distributed Two-Agent MSSR Batch Run on {num_gpus} devices: {gpu_ids}")
    
    print(f"Loading questions data from: {args.annotations_json}")
    with open(args.annotations_json, "r") as file:
        questions_data = json.load(file)
    
    for q in questions_data:
        if "image_paths" in q and len(q["image_paths"]) > 1:
            q["question"] = q["question"] + f'\nTotal number of images: {len(q["image_paths"])}'

    
    total_questions = len(questions_data)
    print(f"Total questions to process: {total_questions}")
    if total_questions < len(gpu_ids):
        gpu_ids = gpu_ids[:total_questions]
        num_gpus = total_questions

    results_folder_path = os.path.join(
        args.results_pth, 
        f"distributed_batch_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.annotations_json.split('/')[-1].split('.')[0]}"
    )
    os.makedirs(results_folder_path)
    print(f"Results will be saved to: {results_folder_path}")
    
    questions_per_gpu = total_questions // num_gpus
    remaining_questions = total_questions % num_gpus
    
    question_subsets = []
    start_idx = 0
    
    for i, gpu_id in enumerate(gpu_ids):
        subset_size = questions_per_gpu + (1 if i < remaining_questions else 0)
        end_idx = start_idx + subset_size
        
        subset = questions_data[start_idx:end_idx]
        question_subsets.append(subset)
        
        print(f"GPU {gpu_id}: {len(subset)} questions (indices {start_idx}-{end_idx-1})")
        start_idx = end_idx
    
    manager = Manager()
    shared_stats = SharedStatistics(manager)
    
    start_time = time.time()
    shared_stats.set_start_time(start_time)
    
    progress_queue = Queue()
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_queue, total_questions, num_gpus, shared_stats)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    print(f"\nstart time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print("starting GPU processing processes...")
    
    processes = []
    
    for i, gpu_id in enumerate(gpu_ids):
        process = Process(
            target=gpu_worker_process,
            args=(gpu_id, question_subsets[i], args, shared_stats, results_folder_path, progress_queue)
        )
        process.start()
        processes.append(process)
    
    print(f"\nStarted {num_gpus} GPU processes. Processing...")
    
    try:
        for process in processes:
            process.join()
        
        progress_queue.put(None)
        progress_thread.join(timeout=1)
        
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n\nall GPU processes completed!")
        print(f"start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"total time: {format_time(total_time)}")
        if total_questions > 0:
            avg_time_per_question = total_time / total_questions
            print(f"average time per question: {avg_time_per_question:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Terminating all processes...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        return
    
    print("\n" + "=" * 20 + " Aggregating Final Results " + "=" * 20)

    avg_time_per_question = total_time / total_questions if total_questions > 0 else 0.0
    
    final_accuracy, final_question_type_stats = shared_stats.get_current_stats()
    
    all_results = list(shared_stats.results_list)
    successful_results = [r for r in all_results if r and r.get('final_answer') is not None]
    
    final_results_path = os.path.join(results_folder_path, "final_distributed_results.json")
    with open(final_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    successful_count = len(successful_results)
    correct_count = shared_stats.correct_count.value
    overall_success_rate = successful_count / total_questions if total_questions > 0 else 0.0
    
    summary_path = os.path.join(results_folder_path, "distributed_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Distributed Two-Agent MSSR Batch Processing Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"GPUs Used: {gpu_ids}\n")
        f.write(f"Total Questions: {total_questions}\n")
        f.write(f"Successfully Processed: {successful_count}\n")
        f.write(f"Success Rate: {overall_success_rate:.4f}\n")
        f.write(f"Accuracy (on successful): {final_accuracy:.4f}\n")
        f.write(f"Overall Accuracy: {correct_count / total_questions:.4f}\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processing time: {format_time(total_time)}\n")
        if total_questions > 0:
            f.write(f"Average time per question: {avg_time_per_question:.2f} seconds\n")
        f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=" * 30 + " Question Type Breakdown " + "=" * 30 + "\n")
        for q_type, stats in sorted(final_question_type_stats.items()):
            type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            f.write(f"{q_type}: {type_accuracy:.4f} ({stats['correct']}/{stats['total']})\n")
    
    print("Distributed batch run completed successfully!")
    print(f"Final summary: {summary_path}")
    print(f"Overall Accuracy: {correct_count}/{total_questions} = {correct_count/total_questions:.4f}")

    print("\n=== Question Type Breakdown ===")
    for q_type, stats in sorted(final_question_type_stats.items()):
        type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{q_type}: {type_accuracy:.4f} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


"""
python src/distributed_batch_runner.py \
    --annotations-json dataset/MMSI-Bench/mmsi_bench.json \
    --model-name gemini-2.5-flash \
    --prompt-mode with-image \
    --gpu-ids "0,1,2,3"

python src/distributed_batch_runner.py \
    --annotations-json dataset/ViewSpatial-Bench/ViewSpatial-Bench_processed.json \
    --model-name gemini-2.5-flash \
    --prompt-mode with-image \
    --gpu-ids "0,1,2,3" \
    --image-pth dataset/ViewSpatial-Bench
"""
