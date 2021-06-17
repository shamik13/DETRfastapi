FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install --no-cache-dir opencv-python-headless torch==1.8.0 torchvision==0.9.0 numpy Pillow matplotlib python-multipart scipy
COPY ./app /app
COPY resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth