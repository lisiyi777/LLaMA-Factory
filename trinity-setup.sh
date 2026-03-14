#conda create -y -n llama-factory python=3.10
#conda activate llama-factory
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install rf100vl
pip install qwen_vl_utils==0.0.8
pip install albumentationsx
pip install -e ".[metrics,deepspeed]" --no-build-isolation
