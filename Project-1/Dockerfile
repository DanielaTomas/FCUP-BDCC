FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --upgrade pip

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

#Define the project ID
ENV GOOGLE_CLOUD_PROJECT="bdcc-project1-417811"

# Expose the port Flask will run on
EXPOSE 8080

# Define the command to run the Flask application
CMD ["python3", "main.py"]

