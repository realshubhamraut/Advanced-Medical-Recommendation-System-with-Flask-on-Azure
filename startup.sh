#!/bin/bash

# Install the necessary system libraries
apt-get update && apt-get install -y libgl1-mesa-glx

# Start the application
gunicorn --bind=0.0.0.0:8000 app:app