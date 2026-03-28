"""
RunPod Serverless Handler for Hunyuan3D-2.1
Converts images (single or multi-view) to 3D models (GLB format).
"""

import os
import sys
import base64
import tempfile
import uuid
import traceback

import torch

# Add Hunyuan3D module paths
REPO_DIR = "/workspace/Hunyuan3D-2.1"
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))

import runpod
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Model config - download from HuggingFace at startup
# ---------------------------------------------------------------------------
MODEL_ID = "tencent/Hunyuan3D-2.1"
MODEL_LOCAL_DIR = "/workspace/models/Hunyuan3D-2.1"


# ---------------------------------------------------------------------------
# Global model references (loaded once when the worker starts)
# ---------------------------------------------------------------------------
shape_pipeline = None
paint_pipeline = None


def download_model():
    """Download model from HuggingFace if not already present."""
    if os.path.isdir(MODEL_LOCAL_DIR) and os.listdir(MODEL_LOCAL_DIR):
        print(f"Model already downloaded at {MODEL_LOCAL_DIR}")
        return
    print(f"Downloading {MODEL_ID} from HuggingFace...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_LOCAL_DIR,
    )
    print("Download complete.")


def load_models():
    """Download and load shape and paint pipelines into GPU memory."""
    global shape_pipeline, paint_pipeline

    download_model()

    print("Loading Hunyuan3D-2.1 shape pipeline...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=False,
        variant="fp16",
        subfolder="hunyuan3d-dit-v2-1",
    )
    shape_pipeline.enable_flashvdm(mc_algo="mc")
    print("Shape pipeline loaded.")

    print("Loading Hunyuan3D-2.1 paint pipeline...")
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_config.realesrgan_ckpt_path = os.path.join(
        REPO_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"
    )
    paint_config.multiview_cfg_path = os.path.join(
        REPO_DIR, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
    )
    paint_config.custom_pipeline = os.path.join(
        REPO_DIR, "hy3dpaint", "hunyuanpaintpbr"
    )
    paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
    print("Paint pipeline loaded.")


def decode_image(image_b64: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def encode_file_b64(file_path: str) -> str:
    """Read a file and return its contents as a base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    RunPod serverless handler.

    Input schema:
    {
        "image": "<base64 encoded image>",         # REQUIRED
        "texture": true/false,                      # default: true
        "remove_background": true/false,            # default: true
        "seed": 1234,                               # default: 1234
        "octree_resolution": 256,                   # 64-512, default: 256
        "num_inference_steps": 30,                  # 1-50, default: 30
        "guidance_scale": 5.0,                      # 0.1-20.0, default: 5.0
        "num_chunks": 8000,                         # 1000-20000, default: 8000
        "face_count": 40000,                        # 1000-100000, default: 40000
        "paint_resolution": 512,                    # 512 or 768, default: 512
        "max_num_view": 6                           # 6-9, default: 6
    }

    Returns:
    {
        "glb_base64": "<base64 encoded GLB file>",
        "format": "glb"
    }
    """
    try:
        job_input = job["input"]

        # --- Validate required input ---
        image_b64 = job_input.get("image")
        if not image_b64:
            return {"error": "No 'image' field provided. Send a base64-encoded image."}

        # --- Parse parameters ---
        do_texture = job_input.get("texture", True)
        remove_bg = job_input.get("remove_background", True)
        seed = job_input.get("seed", 1234)
        octree_resolution = job_input.get("octree_resolution", 256)
        num_inference_steps = job_input.get("num_inference_steps", 30)
        guidance_scale = job_input.get("guidance_scale", 5.0)
        num_chunks = job_input.get("num_chunks", 8000)
        face_count = job_input.get("face_count", 40000)
        paint_resolution = job_input.get("paint_resolution", 512)
        max_num_view = job_input.get("max_num_view", 6)

        # --- Decode input image ---
        image = decode_image(image_b64)

        # --- Optional background removal ---
        if remove_bg:
            from hy3dshape.rembg import BackgroundRemover
            bg_remover = BackgroundRemover()
            image = bg_remover(image)

        # --- Create temp directory for outputs ---
        work_dir = tempfile.mkdtemp(prefix="hunyuan3d_")
        job_id = str(uuid.uuid4())[:8]

        # --- Shape generation ---
        print(f"[{job_id}] Generating 3D shape...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        mesh = shape_pipeline(
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=generator,
            output_type="trimesh",
        )[0]

        untextured_path = os.path.join(work_dir, f"{job_id}_shape.glb")
        mesh.export(untextured_path)
        print(f"[{job_id}] Shape generation complete.")

        # --- Texture generation (optional) ---
        if do_texture:
            print(f"[{job_id}] Generating textures...")

            # Clear some VRAM before texture pass
            torch.cuda.empty_cache()

            textured_path = os.path.join(work_dir, f"{job_id}_textured.glb")

            # Reconfigure paint pipeline if non-default settings
            if paint_resolution != 512 or max_num_view != 6:
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
                paint_config = Hunyuan3DPaintConfig(
                    max_num_view=max_num_view,
                    resolution=paint_resolution,
                )
                paint_config.realesrgan_ckpt_path = os.path.join(
                    REPO_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"
                )
                paint_config.multiview_cfg_path = os.path.join(
                    REPO_DIR, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
                )
                paint_config.custom_pipeline = os.path.join(
                    REPO_DIR, "hy3dpaint", "hunyuanpaintpbr"
                )
                global paint_pipeline
                paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

            paint_pipeline(
                mesh_path=untextured_path,
                image_path=image,
                output_mesh_path=textured_path,
                use_remesh=True,
                save_glb=True,
            )

            output_path = textured_path
            print(f"[{job_id}] Texture generation complete.")
        else:
            output_path = untextured_path

        # --- Encode result ---
        glb_b64 = encode_file_b64(output_path)

        # --- Cleanup temp files ---
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "glb_base64": glb_b64,
            "format": "glb",
            "textured": do_texture,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Load models at worker startup, then start the RunPod serverless handler
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})
