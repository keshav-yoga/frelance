FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium-driver \
    wget \
    gnupg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p logs templates data

# Expose ports
EXPOSE 7860 11434

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
