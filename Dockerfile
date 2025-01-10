# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirment.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirment.txt

# Copy the entire project into the container
COPY . .

# Expose the port that your application will run on (e.g., Flask default port 5000)
EXPOSE 8000

# Define the command to run the application (e.g., using Gunicorn if it's a web app)
# CMD ["python", "app2.py"]
CMD ["uvicorn", "-w", "4", "-b", "0.0.0.0:8000", "app2:app"]

