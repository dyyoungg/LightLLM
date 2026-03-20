pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install --upgrade uv -i https://pypi.tuna.tsinghua.edu.cn/simple/
uv pip install vllm --torch-backend=cu124 --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
cd /mnt/afs/yangdeyu/dependency/lightllm-dev
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd /mnt/afs/yangdeyu/dependency/LightKernel
pip install -v .
pip install numexpr transformers==4.51.1 av imageio pypinyin orjson setproctitle jieba funasr_onnx
pip install /mnt/afs/yangdeyu/dependency/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd /mnt/afs/yangdeyu/dependency/LightTTS
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install /mnt/afs/yangdeyu/dependency/LightTTS/pretrained_models/CosyVoice-ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl
pip install /mnt/afs/yangdeyu/dependency/LightTTS/pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl