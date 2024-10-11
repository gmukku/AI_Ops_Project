# Use a base image with Python
FROM python:3.11-slim

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY . .

# Make the startup script executable
RUN chmod +x startup.sh

# Expose the port your app runs on
EXPOSE 5000

# Run the startup script
CMD ["./startup.sh"]
