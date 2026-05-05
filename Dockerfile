# Dockerfile for ETA Challenge submission.
# Target total image size: <= 2.5 GB.
#
# Build:
#   docker build -t my-eta .
# Test the grader pathway:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only PyTorch to keep image small (~200MB vs ~2GB for full torch)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source modules
COPY features/ ./features/
COPY model/ ./model/

# Copy submission surface + trained weights + artifacts
COPY predict.py grade.py ./
COPY model.pt ./
COPY lgbm_model.txt ./
COPY data/zone_pair_stats/zone_pair_stats.pkl ./data/zone_pair_stats/zone_pair_stats.pkl

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
