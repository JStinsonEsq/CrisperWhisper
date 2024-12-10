FROM python:3.9-slim

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y sox ffmpeg libsndfile1

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . /app
RUN chmod +x /app/serve.sh

ENV HF_TOKEN=HF_TOKEN

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "/app/main.py"] 
