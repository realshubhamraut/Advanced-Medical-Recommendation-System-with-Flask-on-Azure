# Use a slim Python 3.10 base image to keep it lightweight
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies, avoiding cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app will run on (default is 8000 for Gunicorn)
EXPOSE 8000

# Command to run your app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]