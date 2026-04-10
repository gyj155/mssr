# MSSR: Minimal Sufficient Spatial Reasoner

Official implementation of **Pursuing Minimal Sufficiency in Spatial Reasoning** [[arXiv]](https://arxiv.org/abs/2510.16688).

MSSR is a zero-shot, training-free dual-agent framework for multi-view 3D spatial reasoning. It constructs a set of 3D perception results that is both sufficient to answer a spatial query and minimal to avoid reasoning over redundant information.

![alt text](assets/overview.jpg)
![alt text](assets/method2.jpg)
## Setup

### 1. Environment

```bash
conda create -n mssr python=3.10
conda activate mssr
pip install -r requirements.txt
```

### 2. Install Sub-modules

```bash
cd src/models/GroundingDINO && pip install --no-build-isolation -e . && cd ../../..

cd src/models/sam2 && pip install -e . && cd ../../..

cd vggt && pip install -e . && cd ..
```

### 3. Download Model Weights

```bash
bash scripts/download_weights.sh
```

### 4. API Key

We use LLM APIs for code generation and reasoning. Set one of the following environment variables based on your provider:

```bash
# Option 1: Google Gemini (recommended for gemini-* models)
export GOOGLE_API_KEY="your-google-api-key"

# Option 2: OpenAI (for gpt-* models)
export OPENAI_API_KEY="your-openai-api-key"

# Option 3: Custom OpenAI-compatible endpoint
export API_BASE_URL="https://your-endpoint.com/v1"
export API_KEY="your-api-key"
```

### 5. Prepare Datasets

```bash
python scripts/prepare_datasets.py --dataset all
```

This downloads raw data from HuggingFace ([MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench), [ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench)), extracts images, and generates the annotation JSON files.

## Usage

All commands should be run from the **project root** directory.

> **Model selection:** Please use `--model-name gpt-4o` to reproduce the paper results. For development and experiments, we recommend `gemini-3.1-flash-lite-preview`, it is significantly cheaper while achieving comparable performance. In principle, any VLM with an OpenAI-compatible API can be used by setting `API_BASE_URL` and `API_KEY`.

### Interactive Mode

Process individual questions interactively:

```bash
python src/runner_2agents.py \
    --model-name gemini-3.1-flash-lite-preview \
    --annotations-json dataset/MMSI-Bench/mmsi_bench.json \
    --image-pth dataset/MMSI-Bench
```

### Batch Evaluation

Evaluate on the full dataset:

```bash
# Quick test with 10% subset
CUDA_VISIBLE_DEVICES=0 python src/batch_2agent_runner.py \
    --model-name gemini-3.1-flash-lite-preview \
    --subset-ratio 0.1

# Full MMSI-Bench evaluation
CUDA_VISIBLE_DEVICES=0 python src/batch_2agent_runner.py \
    --model-name gemini-3.1-flash-lite-preview \
    --annotations-json dataset/MMSI-Bench/mmsi_bench.json \
    --image-pth dataset/MMSI-Bench

# ViewSpatial-Bench evaluation
CUDA_VISIBLE_DEVICES=0 python src/batch_2agent_runner.py \
    --model-name gemini-3.1-flash-lite-preview \
    --annotations-json dataset/ViewSpatial-Bench/ViewSpatial-Bench_processed.json \
    --image-pth dataset/ViewSpatial-Bench
```

### Use multiple gpus (one process per gpu)

```bash
python src/distributed_batch_runner.py \
    --model-name gemini-3.1-flash-lite-preview \
    --gpu-ids "0,1,2,3" \
    --annotations-json dataset/MMSI-Bench/mmsi_bench.json \
    --image-pth dataset/MMSI-Bench
```

## Visualization

A Flask-based web tool is provided for inspecting results:

```bash
python dual_agent_viewer.py
```

Then open `http://localhost:5000` in your browser. You can also specify a custom results directory:

```bash
DUAL_AGENT_RESULTS_PATH=/path/to/results python dual_agent_viewer.py
```

The viewer allows browsing:
- PA-generated code and execution traces per iteration
- RA reasoning, information curation, and decision logs
- MSS evolution across iterations
- Per-question correctness and statistics

## Acknowledgements
We would like to thank the following works for their contributions to the community and our codebase:
* [VADAR](https://github.com/damianomarsili/VADAR)
* [VGGT](https://github.com/facebookresearch/vggt)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* [SAM2](https://github.com/facebookresearch/sam2)

## Citation

```bibtex
@article{guo2025pursuing,
  title={Pursuing Minimal Sufficiency in Spatial Reasoning},
  author={Guo, Yejie and Hou, Yunzhong and Ma, Wufei and Tang, Meng and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2510.16688},
  year={2025}
}
```