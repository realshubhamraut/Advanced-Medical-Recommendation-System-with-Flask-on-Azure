#!/bin/bash

# Install the necessary system libraries
apt-get update && apt-get install -y libgl1-mesa-glx
apt-get update && apt-get install -y libglib2.0-0


# Start the application
gunicorn --bind=0.0.0.0:8000 app:app