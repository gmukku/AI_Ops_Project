#!/bin/bash

# Make the startup script executable
chmod +x startup.sh

# Start the Flask app in the background
python prediction.py &

# Loop to continuously check and call the endpoints
while true; do
    # Call the prediction endpoint
    curl http://localhost:5000/predict
    sleep 10  # Wait for 10 seconds before the next call
    
    # Call the report endpoint
    curl http://localhost:5000/report
    sleep 10  # Wait for 10 seconds before the next call
done
