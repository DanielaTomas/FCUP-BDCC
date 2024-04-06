# Use the official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port Flask will run on
EXPOSE 8080

# Define the command to run the Flask application
CMD ["python", "main.py"]