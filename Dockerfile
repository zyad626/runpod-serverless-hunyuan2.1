# =============================================================================
# RunPod Serverless Worker for Hunyuan3D-2.1
# Image/Multi-view -> 3D Model (GLB) generation
#
# Multi-stage build:
#   Stage 1 (builder) - CUDA devel image for compiling extensions
#   Stage 2 (runtime) - CUDA runtime image, much smaller final output
# =============================================================================

# ========================== STAGE 1: BUILD ==================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
# Only target modern GPUs commonly used on RunPod (Ampere, Ada, Hopper)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget unzip cmake \
    pkg-config libeigen3-dev python3-dev python3-setuptools \
    pybind11-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge, no Anaconda TOS)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3 && \
    rm Miniforge3-Linux-x86_64.sh

ENV PATH="/workspace/miniforge3/bin:${PATH}"
RUN conda create -n hunyuan3d python=3.10 -y
ENV PATH="/workspace/miniforge3/envs/hunyuan3d/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=hunyuan3d

# Build tools
RUN conda install -n hunyuan3d ninja -y && \
    conda install -n hunyuan3d cuda -c nvidia/label/cuda-12.4.1 -y && \
    conda install -n hunyuan3d libstdcxx-ng -y && \
    conda clean -afy

# PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone repo
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git

# Python dependencies
COPY builder/requirements.txt /tmp/requirements.txt
# Remove bpy (Blender) and deepspeed from upstream requirements - not available as pip wheels
# and not needed for serverless inference
RUN sed -i '/^bpy/d; /^deepspeed/d' /workspace/Hunyuan3D-2.1/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /workspace/Hunyuan3D-2.1/requirements.txt

# Build custom CUDA extensions
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
    pip install --no-build-isolation -e .

RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh

# Download RealESRGAN weights
RUN mkdir -p /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt/

# Apply path fixes
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
    sed -i 's|self\.multiview_cfg_path = "cfgs/hunyuan-paint-pbr\.yaml"|self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"|' \
    textureGenPipeline.py

RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/utils && \
    sed -i 's|custom_pipeline = config\.custom_pipeline|custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")|' \
    multiview_utils.py

# =============================================================================
# Clean up build artifacts to reduce size before copy
# =============================================================================
RUN conda clean -afy && \
    pip cache purge && \
    find /workspace/miniforge3 -name '*.a' -delete && \
    find /workspace/miniforge3 -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /workspace/Hunyuan3D-2.1/.git && \
    rm -rf /workspace/Hunyuan3D-2.1/assets && \
    rm -rf /workspace/Hunyuan3D-2.1/docker && \
    rm -rf /tmp/*


# ========================== STAGE 2: RUNTIME ================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# Runtime-only system dependencies (no build-essential, no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libegl1-mesa libgles2-mesa \
    libxrender1 libglib2.0-0 \
    libxi6 libsm6 libxext6 \
    libxkbcommon-x11-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire conda environment and repo from builder
COPY --from=builder /workspace/miniforge3 /workspace/miniforge3
COPY --from=builder /workspace/Hunyuan3D-2.1 /workspace/Hunyuan3D-2.1

ENV PATH="/workspace/miniforge3/envs/hunyuan3d/bin:/workspace/miniforge3/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=hunyuan3d

# Copy handler
COPY src/handler.py /workspace/handler.py

WORKDIR /workspace/Hunyuan3D-2.1

CMD ["python", "-u", "/workspace/handler.py"]
