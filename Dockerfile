# Stable, slim Python base recommended by RunPod docs
# https://docs.runpod.io/serverless/workers/custom-worker
FROM paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9

WORKDIR /

COPY . .

RUN pip install --no-cache-dir \
      paddleocr>=3.1.0 \
      PyMuPDF>=1.23.0 \
      requests>=2.31.0

CMD ["python3", "-u", "rp_handler.py"]
