# Use slim Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy code
COPY . .


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
