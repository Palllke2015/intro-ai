# Use slim Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run without reload (production)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
