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

# use the CPU version 
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    torchaudio==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies    
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ .

# Expose the port
EXPOSE 8082

# Run the app
ENTRYPOINT ["uvicorn"]
CMD ["orchestrator-server:app", "--host", "0.0.0.0", "--port", "8082"]
