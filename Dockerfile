FROM python:3.12-slim

WORKDIR /app

# Install system dependency required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application and trained model
COPY src/ ./src/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
