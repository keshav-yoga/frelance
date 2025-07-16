#!/bin/bash

# Start Ollama in background
ollama serve &

# Wait for Ollama to start
sleep 10

# Pull required models
ollama pull mistral
ollama pull codellama

# Start the application
python freelance_automation_final.py
