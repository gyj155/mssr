# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import json

# Set Gradio cache directory to be local
gradio_tmp_dir = os.path.join(os.path.dirname(__file__), "gradio_tmp")
os.makedirs(gradio_tmp_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = gradio_tmp_dir

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(model_dir, exist_ok=True)
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, model_dir=model_dir))

model.eval()
model = model.to(device)

# Load MMSI-Bench dataset
DATASET_PATH = "dataset/MMSI-Bench/mmsi_bench.json"
IMAGES_DIR = "dataset/MMSI-Bench/images"

print("Loading MMSI-Bench dataset...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    mmsi_dataset = json.load(f)

print(f"Loaded {len(mmsi_dataset)} questions from MMSI-Bench dataset")

# -------------------------------------------------------------------------
# Core model inference (same as original)
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    print(f"World points shape: {world_points.shape}, type: {type(world_points)}")
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# Load images from MMSI-Bench dataset by question ID
# -------------------------------------------------------------------------
def load_images_by_id(question_id):
    """
    Load images for a specific question ID from the MMSI-Bench dataset.
    Returns (target_dir, image_paths, question_info).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Find the question by ID
    question_data = None
    for q in mmsi_dataset:
        if q["id"] == question_id:
            question_data = q
            break
    
    if question_data is None:
        raise ValueError(f"Question ID {question_id} not found in dataset")

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(os.path.dirname(__file__), f"inputs/mmsi_question_{question_id}_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []
    
    # Copy images from the dataset
    for i, rel_path in enumerate(question_data["image_paths"]):
        # Convert relative path to absolute path
        # rel_path is like "./images/0_0.jpg", we need "dataset/MMSI-Bench/images/0_0.jpg"
        img_filename = os.path.basename(rel_path)
        src_path = os.path.join(IMAGES_DIR, img_filename)
        dst_path = os.path.join(target_dir_images, f"{i:06d}.jpg")
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            image_paths.append(dst_path)
        else:
            print(f"Warning: Image {src_path} not found")

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Loaded {len(image_paths)} images for question {question_id}; took {end_time - start_time:.3f} seconds")
    
    return target_dir, image_paths, question_data


# -------------------------------------------------------------------------
# Update display when question ID changes
# -------------------------------------------------------------------------
def update_question_display(question_id):
    """
    Load images and question info when question ID changes.
    """
    if question_id is None or question_id < 0 or question_id >= len(mmsi_dataset):
        return None, None, "Please select a valid question ID (0-999)", None
    
    try:
        target_dir, image_paths, question_data = load_images_by_id(question_id)
        
        # Format question info
        question_info = f"""
**Question ID:** {question_data['id']}
**Question Type:** {question_data['question_type']}
**Image Count:** {question_data['image_count']}

**Question:** {question_data['question']}

**Answer:** {question_data['answer']}
**Reasoning:** {question_data['thought']}
        """
        
        return target_dir, image_paths, "Images loaded successfully. Click 'Reconstruct' to begin 3D processing.", question_info
        
    except Exception as e:
        return None, None, f"Error loading question {question_id}: {str(e)}", None


# -------------------------------------------------------------------------
# Reconstruction function (similar to original but adapted for MMSI data)
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=1.0,  # Default to 1% as requested
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Depthmap and Camera Branch",  # Default as requested
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please select a question ID first.", None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Navigation functions for Previous/Next buttons
# -------------------------------------------------------------------------
def go_to_previous_question(current_id):
    """Go to the previous question ID."""
    if current_id is None:
        return 0
    new_id = max(0, current_id - 1)
    return new_id


def go_to_next_question(current_id):
    """Go to the next question ID."""
    if current_id is None:
        return 0
    new_id = min(999, current_id + 1)
    return new_id


# -------------------------------------------------------------------------
# Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Soft()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .question-info {
        font-size: 14px !important;
        background-color: #f8f9fa;
        color: #212529 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Dark mode support for question info */
    @media (prefers-color-scheme: dark) {
        .question-info {
            background-color: #2d3748 !important;
            color: #e2e8f0 !important;
            border-left: 4px solid #60a5fa !important;
        }
    }
    
    /* Navigation button styles */
    .nav-button {
        min-height: 40px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .nav-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Hidden state for tracking
    is_example = gr.Textbox(label="is_example", visible=False, value="False")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <h2>📊 MMSI-Bench Dataset Demo</h2>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>This demo allows you to run VGGT on questions from the MMSI-Bench dataset. Select a question ID (0-999) to automatically load the corresponding images and perform 3D reconstruction.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Select Question ID:</strong> Choose a question ID from 0 to 999 using the number input below.</li>
        <li><strong>Preview:</strong> The question details and images will appear automatically.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer. You can rotate, pan, and zoom to explore the model.</li>
        <li><strong>Adjust Visualization (Optional):</strong> Use the controls below to fine-tune the visualization.</li>
    </ol>
    <p><strong style="color: #0ea5e9;">Note:</strong> <span style="color: #0ea5e9; font-weight: bold;">Default settings: Depthmap and Camera Branch, 1% confidence threshold.</span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            # Question ID input with navigation buttons
            with gr.Row():
                prev_btn = gr.Button("◀ Previous", scale=1, variant="secondary", elem_classes=["nav-button"])
                question_id = gr.Number(
                    label="Question ID (0-999)", 
                    value=0, 
                    minimum=0, 
                    maximum=999, 
                    step=1,
                    interactive=True,
                    scale=2
                )
                next_btn = gr.Button("Next ▶", scale=1, variant="secondary", elem_classes=["nav-button"])
            
            # Question information display
            question_info = gr.Markdown(
                "Select a question ID to see details here.", 
                elem_classes=["question-info"]
            )

            # Image gallery
            image_gallery = gr.Gallery(
                label="Preview Images",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please select a question ID, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",  # Default as requested
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=1.0, step=0.1, label="Confidence Threshold (%)")  # Default 1% as requested
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------
    
    # Update question display when ID changes
    question_id.change(
        fn=update_question_display,
        inputs=[question_id],
        outputs=[target_dir_output, image_gallery, log_output, question_info],
    )
    
    # Previous button logic
    prev_btn.click(
        fn=go_to_previous_question,
        inputs=[question_id],
        outputs=[question_id],
    )
    
    # Next button logic
    next_btn.click(
        fn=go_to_next_question,
        inputs=[question_id],
        outputs=[question_id],
    )

    # "Reconstruct" button logic
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    )

    # Real-time Visualization Updates
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)

"""
Usage:
CUDA_VISIBLE_DEVICES=7 python proj/vggt/demo_mmsi_bench.py
"""