# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Create output directory structure
RUN mkdir -p /app/outputs/models /app/outputs/results

# Copy the requirements file and model training script into the container
COPY src/ .

# Install dependencies
RUN pip install -r requirements.txt

# Run the script when the container launches
CMD ["python", "main.py"]