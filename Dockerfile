FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (7860 for HF Spaces)
EXPOSE 8000
EXPOSE 7860

# Ensure start script is executable
RUN chmod +x start.sh

# Run the start script
CMD ["./start.sh"]
