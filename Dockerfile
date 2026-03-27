# =============================================================================
# RunPod Serverless Worker for Hunyuan3D-2.1
# Image/Multi-view -> 3D Model (GLB) generation
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# =============================================================================
# System dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget vim unzip git-lfs curl cmake \
    pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    mesa-utils-extra libxrender1 libeigen3-dev \
    python3-dev python3-setuptools libcgal-dev \
    libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Miniforge (conda-forge, no Anaconda TOS) + Python 3.10
# =============================================================================
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3 && \
    rm Miniforge3-Linux-x86_64.sh

ENV PATH="/workspace/miniforge3/bin:${PATH}"
RUN conda create -n hunyuan3d python=3.10 -y
ENV PATH="/workspace/miniforge3/envs/hunyuan3d/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=hunyuan3d

# Build tools via conda
RUN conda install -n hunyuan3d ninja -y && \
    conda install -n hunyuan3d cuda -c nvidia/label/cuda-12.4.1 -y && \
    conda install -n hunyuan3d libstdcxx-ng -y

# =============================================================================
# PyTorch (CUDA 12.4)
# =============================================================================
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# Clone Hunyuan3D-2.1 repository
# =============================================================================
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git

# =============================================================================
# Install Python dependencies
# =============================================================================
COPY builder/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /workspace/Hunyuan3D-2.1/requirements.txt

# =============================================================================
# Build custom CUDA extensions
# =============================================================================

# 1. Custom rasterizer
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
    CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
    pip install -e .

# 2. DifferentiableRenderer mesh_inpaint_processor
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh

# =============================================================================
# Download RealESRGAN weights
# =============================================================================
RUN mkdir -p /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt/

# =============================================================================
# Apply path fixes for running from repo root
# =============================================================================
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
    sed -i 's|self\.multiview_cfg_path = "cfgs/hunyuan-paint-pbr\.yaml"|self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"|' \
    textureGenPipeline.py

RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/utils && \
    sed -i 's|custom_pipeline = config\.custom_pipeline|custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")|' \
    multiview_utils.py

# =============================================================================
# Copy handler code
# =============================================================================
COPY src/handler.py /workspace/handler.py

WORKDIR /workspace/Hunyuan3D-2.1

# =============================================================================
# Start the RunPod serverless worker
# =============================================================================
CMD ["python", "-u", "/workspace/handler.py"]
