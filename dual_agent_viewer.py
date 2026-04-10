#!/usr/bin/env python3
"""
Dual-Agent Results Viewer - A web tool for visualizing Dual-Agent experiment results
"""

import os
import json
import mimetypes
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from typing import Dict, List, Optional, Union

app = Flask(__name__)

# Configuration
def load_config():
    """Load configuration from config.json or environment variables"""
    default_config = {
        "paths": {
            "results_root": os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "results")
        }
    }
    

    
    # Fall back to environment variables
    results_path = os.environ.get('DUAL_AGENT_RESULTS_PATH')
    if results_path:
        default_config["paths"]["results_root"] = results_path
    
    return default_config

config = load_config()
RESULTS_ROOT = config["paths"]["results_root"]

class DualAgentResultsExplorer:
    """Helper class to explore and analyze Dual-Agent results directories"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
    
    def _extract_numeric_id(self, text: str) -> int:
        """Extract numeric ID from text for proper sorting"""
        try:
            # Try to extract numeric part from question_* format
            if "question_" in text:
                numeric_part = text.split("question_")[-1]
                return int(numeric_part)
            # For other formats, try to extract trailing numbers
            import re
            match = re.search(r'(\d+)$', text)
            if match:
                return int(match.group(1))
            return 0
        except (ValueError, IndexError):
            return 0
    
    def _sort_items_by_question_id(self, items):
        """Sort items with question IDs numerically"""
        def sort_key(item):
            name = item.name if hasattr(item, 'name') else item.get('name', '')
            if name.startswith('question_'):
                return (0, self._extract_numeric_id(name))  # Questions first
            else:
                return (1, name)  # Other items second, alphabetically
        return sorted(items, key=sort_key)
    
    def get_directory_listing(self, path: str = "") -> Dict:
        """Get directory listing with metadata about result types"""
        current_path = self.root_path / path if path else self.root_path
        
        if not current_path.exists():
            return {"error": "Path does not exist"}
        
        items = []
        
        # Check if current directory is a distributed batch
        result_type = self._detect_result_type(current_path)
        if result_type == "distributed_batch":
            # For distributed batch, aggregate all questions from all GPU directories
            all_questions = []
            gpu_dirs = [d for d in current_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")]
            
            for gpu_dir in gpu_dirs:
                question_dirs = [d for d in gpu_dir.iterdir() if d.is_dir() and d.name.startswith("question_")]
                for question_dir in question_dirs:
                    # Add question info with modified path to include gpu directory
                    question_info = self._get_question_result_info(question_dir)
                    question_item = {
                        "name": question_dir.name,
                        "type": "directory", 
                        "result_type": "question",
                        "has_exec_env": self._has_exec_env(question_dir),
                        "is_question_dir": True,
                        "is_correct": question_info["is_correct"],
                        "predicted_answer": question_info["predicted_answer"],
                        "ground_truth": question_info["ground_truth"],
                        "question_type": question_info["question_type"],
                        "execution_successful": question_info["execution_successful"],
                        "gpu_name": gpu_dir.name,  # Keep track of which GPU for path resolution
                        "full_path": str(question_dir.relative_to(self.root_path))  # Store full path
                    }
                    all_questions.append(question_item)
            
            # Sort all questions by numeric ID
            items = self._sort_items_by_question_id(all_questions)
            
            # Also add non-gpu files in the distributed batch directory
            for item in current_path.iterdir():
                if not (item.is_dir() and item.name.startswith("gpu_")):
                    if item.is_dir():
                        item_result_type = self._detect_result_type(item)
                        items.append({
                            "name": item.name,
                            "type": "directory",
                            "result_type": item_result_type,
                            "has_exec_env": self._has_exec_env(item),
                            "is_question_dir": False
                        })
                    else:
                        items.append({
                            "name": item.name,
                            "type": "file", 
                            "size": item.stat().st_size,
                            "is_question_dir": False
                        })
            
        else:
            # Normal directory processing
            sorted_items = self._sort_items_by_question_id(list(current_path.iterdir()))
            for item in sorted_items:
                if item.is_dir():
                    # Check if this is a results directory
                    item_result_type = self._detect_result_type(item)
                    item_info = {
                        "name": item.name,
                        "type": "directory",
                        "result_type": item_result_type,
                        "has_exec_env": self._has_exec_env(item),
                        "question_count": self._count_questions(item) if item_result_type in ["batch", "distributed_batch"] else None
                    }
                    
                    # If this is a question directory, add correctness info
                    if item.name.startswith("question_"):
                        question_info = self._get_question_result_info(item)
                        item_info.update({
                            "is_question_dir": True,
                            "is_correct": question_info["is_correct"],
                            "predicted_answer": question_info["predicted_answer"],
                            "ground_truth": question_info["ground_truth"],
                            "question_type": question_info["question_type"],
                            "execution_successful": question_info["execution_successful"]
                        })
                    else:
                        item_info["is_question_dir"] = False
                    
                    items.append(item_info)
                else:
                    items.append({
                        "name": item.name,
                        "type": "file",
                        "size": item.stat().st_size,
                        "is_question_dir": False
                    })
        
        return {
            "current_path": str(current_path.relative_to(self.root_path)),
            "items": items,
            "parent": str(current_path.parent.relative_to(self.root_path)) if current_path != self.root_path else None,
            "is_distributed_batch": result_type == "distributed_batch"
        }
    
    def _detect_result_type(self, path: Path) -> Optional[str]:
        """Detect if directory contains single, batch, or distributed batch results"""
        if not path.is_dir():
            return None
        
        # Check for distributed batch pattern (distributed_batch_* name or contains gpu_* subdirs)
        if path.name.startswith("distributed_batch_"):
            return "distributed_batch"
        gpu_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("gpu_")]
        if gpu_dirs:
            # Verify that gpu dirs contain question dirs
            for gpu_dir in gpu_dirs[:3]:  # Check first few gpu dirs
                question_dirs = [d for d in gpu_dir.iterdir() if d.is_dir() and d.name.startswith("question_")]
                if question_dirs:
                    return "distributed_batch"
        
        # Check for batch pattern (batch_* name or contains question_* subdirs)
        if path.name.startswith("batch_"):
            return "batch"
        question_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("question_")]
        if question_dirs:
            return "batch"
        
        # Check for iteration_* pattern (multi-round agent)
        iteration_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
        if iteration_dirs:
            return "single"

        # Check for single pattern (2agents_* name or contains two_agent_execution dir)
        if path.name.startswith("two_agents_"):
            return "single"
        if (path / "two_agent_execution").exists():
            return "single"

        # Legacy support for old format
        if (path / "program_execution").exists():
            return "single"

        return None
    
    def _has_exec_env(self, path: Path) -> bool:
        """Check if directory has executable environment files"""
        # Check for iteration_* format (multi-round agent)
        iteration_dirs = list(path.rglob("iteration_*"))
        if iteration_dirs:
            return True
        # Check for two_agent_execution format
        two_agent_execs = list(path.rglob("two_agent_execution"))
        if two_agent_execs:
            return True
        # Legacy support for old format
        exec_envs = list(path.rglob("exec_env"))
        return len(exec_envs) > 0
    
    def _count_questions(self, path: Path) -> int:
        """Count number of questions in batch directory or distributed batch"""
        if not path.is_dir():
            return 0
        
        result_type = self._detect_result_type(path)
        
        if result_type == "distributed_batch":
            # For distributed batch, count questions across all GPU directories
            total_questions = 0
            gpu_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("gpu_")]
            for gpu_dir in gpu_dirs:
                question_dirs = [d for d in gpu_dir.iterdir() if d.is_dir() and d.name.startswith("question_")]
                total_questions += len(question_dirs)
            return total_questions
        else:
            # For normal batch, count questions directly
            question_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("question_")]
            return len(question_dirs)
    
    def _get_question_result_info(self, question_path: Path) -> Dict:
        """Get question result info including correctness"""
        result_info = {
            "is_correct": None,
            "predicted_answer": None,
            "ground_truth": None,
            "question_type": None,
            "execution_successful": None
        }
        
        # Try to read question_result.json
        question_result_file = question_path / "question_result.json"
        if question_result_file.exists():
            try:
                with open(question_result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    result_info.update({
                        "is_correct": data.get("is_correct"),
                        "predicted_answer": data.get("predicted_answer"),
                        "ground_truth": data.get("ground_truth"),
                        "question_type": data.get("question_type"),
                        "execution_successful": data.get("execution_successful")
                    })
            except (json.JSONDecodeError, IOError):
                pass
        
        return result_info
    
    def find_exec_envs(self, path: str) -> List[Dict]:
        """Find all execution environment directories in the given path"""
        current_path = self.root_path / path
        exec_envs = []

        # Check if this is a distributed batch directory
        result_type = self._detect_result_type(current_path)
        if result_type == "distributed_batch":
            # For distributed batch, search within all GPU directories
            gpu_dirs = [d for d in current_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")]
            search_paths = gpu_dirs
        else:
            # For normal directories, search within current path
            search_paths = [current_path]

        for search_path in search_paths:
            # Find iteration_* directories (multi-round agent format)
            for iter_path in sorted(search_path.rglob("iteration_*")):
                if not iter_path.is_dir():
                    continue
                # Only match direct iteration dirs (not nested)
                if not iter_path.name.startswith("iteration_"):
                    continue

                question_id = self._extract_question_id(iter_path)
                iteration_num = iter_path.name.replace("iteration_", "")

                # Find analysis HTML file
                analysis_html = None
                for html_file in iter_path.glob("analysis_question_*.html"):
                    analysis_html = html_file
                    break

                files = {
                    "executable_program": iter_path / "executable_program.py",
                    "result_json": iter_path / "result.json",
                    "trace_html": iter_path / "trace.html",
                    "analysis_html": analysis_html
                }
                files_status = {name: fp.exists() if fp else False for name, fp in files.items()}

                # Get question info from parent directory (question_*)
                question_dir = iter_path.parent
                question_info = self._get_question_result_info(question_dir)

                exec_envs.append({
                    "question_id": question_id,
                    "iteration": int(iteration_num) if iteration_num.isdigit() else 0,
                    "path": str(iter_path.relative_to(self.root_path)),
                    "files": files_status,
                    "analysis_html_name": analysis_html.name if analysis_html else None,
                    "complete": files_status["executable_program"] and files_status["result_json"] and files_status["trace_html"],
                    "is_correct": question_info["is_correct"],
                    "predicted_answer": question_info["predicted_answer"],
                    "ground_truth": question_info["ground_truth"],
                    "question_type": question_info["question_type"],
                    "execution_successful": question_info["execution_successful"]
                })

            # Find two_agent_execution directories
            for exec_env_path in search_path.rglob("two_agent_execution"):
                # Get the question identifier
                question_id = self._extract_question_id(exec_env_path)

                # Find analysis HTML file
                analysis_html = None
                for html_file in exec_env_path.glob("analysis_question_*.html"):
                    analysis_html = html_file
                    break

                # Check for required files
                files = {
                    "executable_program": exec_env_path / "executable_program.py",
                    "result_json": exec_env_path / "result.json",
                    "trace_html": exec_env_path / "trace.html",
                    "analysis_html": analysis_html
                }

                files_status = {name: file_path.exists() if file_path else False for name, file_path in files.items()}

                # Get question info from parent directory (question_*)
                question_dir = exec_env_path.parent
                question_info = self._get_question_result_info(question_dir)

                exec_envs.append({
                    "question_id": question_id,
                    "path": str(exec_env_path.relative_to(self.root_path)),
                    "files": files_status,
                    "analysis_html_name": analysis_html.name if analysis_html else None,
                    "complete": files_status["executable_program"] and files_status["result_json"] and files_status["trace_html"],
                    "is_correct": question_info["is_correct"],
                    "predicted_answer": question_info["predicted_answer"],
                    "ground_truth": question_info["ground_truth"],
                    "question_type": question_info["question_type"],
                    "execution_successful": question_info["execution_successful"]
                })

            # Legacy support: Find old exec_env directories
            for exec_env_path in search_path.rglob("program_execution"):
                # Get the question identifier
                question_id = self._extract_question_id(exec_env_path)

                # Check for required files
                files = {
                    "executable_program": exec_env_path / "executable_program.py",
                    "result_json": exec_env_path / "result.json",
                    "trace_html": exec_env_path / "trace.html",
                    "analysis_html": None
                }

                files_status = {name: file_path.exists() if file_path else False for name, file_path in files.items()}

                # Get question info from parent directory (question_*)
                question_dir = exec_env_path.parent
                question_info = self._get_question_result_info(question_dir)

                exec_envs.append({
                    "question_id": question_id,
                    "path": str(exec_env_path.relative_to(self.root_path)),
                    "files": files_status,
                    "analysis_html_name": None,
                    "complete": files_status["executable_program"] and files_status["result_json"] and files_status["trace_html"],
                    "is_correct": question_info["is_correct"],
                    "predicted_answer": question_info["predicted_answer"],
                    "ground_truth": question_info["ground_truth"],
                    "question_type": question_info["question_type"],
                    "execution_successful": question_info["execution_successful"]
                })
        
        # Sort exec_envs by numeric question ID, then by iteration number
        return sorted(exec_envs, key=lambda x: (self._extract_numeric_id(x["question_id"]), x.get("iteration", 0)))
    
    def _extract_question_id(self, exec_env_path: Path) -> str:
        """Extract question ID from execution environment path"""
        parts = exec_env_path.parts
        
        # Look for question_* in path parts
        for part in parts:
            if part.startswith("question_"):
                return part.replace("question_", "")
        
        # For single result format (2agents_*), try to extract from directory name
        for part in parts:
            if part.startswith("two_agents_") and "question_" in part:
                # Extract question ID from format like "2agents_2025-08-08_18-00-36_question_688"
                question_part = part.split("question_")[-1]
                return question_part
        for part in parts:
            if "question_" in part:
                # Extract question ID from format like "program_execution_2025-08-08_18-00-36_question_688"
                question_part = part.split("question_")[-1]
                return question_part
        
        return "unknown"
    
    def read_file_content(self, file_path: str) -> Dict:
        """Read and return file content with appropriate formatting"""
        full_path = self.root_path / file_path
        
        if not full_path.exists():
            return {"error": "File does not exist"}
        
        try:
            if full_path.suffix == ".json":
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                return {"type": "json", "content": content}
            
            elif full_path.suffix == ".py":
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"type": "python", "content": content}
            
            elif full_path.suffix == ".html":
                # For HTML files, we'll serve them directly via iframe
                return {"type": "html", "path": file_path, "filename": full_path.name}
            
            else:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"type": "text", "content": content}
                
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}

# Initialize explorer
explorer = DualAgentResultsExplorer(RESULTS_ROOT)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/browse')
def browse():
    """Browse directory API"""
    path = request.args.get('path', '')
    # Handle distributed batch question navigation
    # If path indicates we're navigating to a question in a distributed batch
    # we may need to resolve the full path
    result = explorer.get_directory_listing(path)
    return jsonify(result)

@app.route('/api/exec_envs')
def get_exec_envs():
    """Get all exec_env directories in a path"""
    path = request.args.get('path', '')
    return jsonify(explorer.find_exec_envs(path))

@app.route('/api/file')
def get_file():
    """Get file content"""
    file_path = request.args.get('path', '')
    full_path = (Path(RESULTS_ROOT) / file_path).resolve()
    if not str(full_path).startswith(str(Path(RESULTS_ROOT).resolve())):
        return jsonify({"error": "Access denied"}), 403
    return jsonify(explorer.read_file_content(file_path))

@app.route('/api/serve_html')
def serve_html():
    """Serve HTML files with enhanced styling"""
    file_path = request.args.get('path', '')
    full_path = (Path(RESULTS_ROOT) / file_path).resolve()

    if not str(full_path).startswith(str(Path(RESULTS_ROOT).resolve())):
        return "Access denied", 403
    if not full_path.exists() or full_path.suffix != '.html':
        return "File not found", 404
    
    # Read the HTML content and inject custom CSS
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Enhanced CSS for better readability
        enhanced_css = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            line-height: 1.8 !important;
            color: #333 !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 20px !important;
            word-wrap: break-word !important;
            background: #ffffff !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            font-weight: 600 !important;
        }
        
        p {
            margin-bottom: 1rem !important;
            text-align: justify !important;
            font-size: 15px !important;
        }
        
        pre {
            background: #f8f9fa !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            overflow-x: auto !important;
            border: 1px solid #e9ecef !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
        }
        
        code {
            background: #f8f9fa !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            font-size: 13px !important;
        }
        
        table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 1rem 0 !important;
        }
        
        th, td {
            border: 1px solid #dee2e6 !important;
            padding: 8px 12px !important;
            text-align: left !important;
        }
        
        th {
            background-color: #f8f9fa !important;
            font-weight: 600 !important;
        }
        
        .traceback, .error {
            background: #ffebee !important;
            border: 1px solid #ffcdd2 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            margin: 1rem 0 !important;
        }
        
        .highlight {
            background: #fff3cd !important;
            border: 1px solid #ffeaa7 !important;
            padding: 0.5rem !important;
            border-radius: 4px !important;
        }
        
        /* For trace.html specific content */
        .step, .execution-step {
            margin: 1rem 0 !important;
            padding: 1rem !important;
            background: #f8f9fa !important;
            border-left: 4px solid #6c7ce7 !important;
            border-radius: 0 8px 8px 0 !important;
        }
        
        .variable, .output {
            background: #e8f5e8 !important;
            padding: 0.5rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 4px !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        }
        
        /* Improve text readability */
        * {
            text-rendering: optimizeLegibility !important;
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
        }
        </style>
        """
        
        # Insert the CSS right after the <head> tag or before </head>
        if '<head>' in content:
            content = content.replace('<head>', '<head>' + enhanced_css)
        elif '</head>' in content:
            content = content.replace('</head>', enhanced_css + '</head>')
        else:
            # If no head tag, add it at the beginning
            content = enhanced_css + content
        
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        return f"Error reading file: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)