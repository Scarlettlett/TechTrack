# Use an official Python runtime as a base image
FROM python:3.9-slim

# Install FFmpeg, OpenCV, and other dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip install opencv-python-headless numpy

# Set up a working directory
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the UDP port for video streaming
EXPOSE 23000

# Run the Inference Service with the streaming setup
CMD ["python", "preprocessing.py"]
