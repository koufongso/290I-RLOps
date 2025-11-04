FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for Box2D + gymnasium
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ .

# Expose the port
EXPOSE 8080

# Run the app
ENTRYPOINT ["uvicorn"]
CMD ["simulator_server:app", "--host", "0.0.0.0", "--port", "8080"]