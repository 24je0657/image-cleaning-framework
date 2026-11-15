FROM python:3.12-slim

# Hugging Face Spaces run as uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 TORCH_HOME=/home/user/.cache/torch
WORKDIR /home/user/app

RUN chown user:user /home/user/app

COPY --chown=user requirements-docker.txt .
USER user
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt

# Bake ResNet50 weights into the image so startup needs no download.
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"

COPY --chown=user . .

ENV IMAGE_CLEANING_JOBS_ROOT=/home/user/jobs \
    IMAGE_CLEANING_MAX_UPLOAD_MB=200 \
    IMAGE_CLEANING_MAX_UNCOMPRESSED_MB=600 \
    IMAGE_CLEANING_MAX_IMAGES=2000 \
    IMAGE_CLEANING_MAX_ZIP_ENTRIES=4000 \
    IMAGE_CLEANING_MAX_CONCURRENT_RUNS=1 \
    IMAGE_CLEANING_MIN_FREE_MB=1024

EXPOSE 7860
CMD ["bash", "scripts/start.sh"]