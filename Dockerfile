# Use the official Python 3.12.9 slim image
FROM python:3.11.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run the application

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0",  "--server.headless", "true", "--server.runOnSave", "true"]
