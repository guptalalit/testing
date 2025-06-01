# Use a lightweight and stable Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        default-jdk \
        poppler-utils \
        ghostscript \
        tesseract-ocr \
        python3-pip \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copy project folders and main files
COPY memory/ ./memory/
COPY static/ ./static/
COPY tasks/ ./tasks/
COPY templates/ ./templates/
COPY Thyroid/ ./Thyroid/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY validate/ ./validate/
# Copy required files
COPY session_id.txt ./
COPY app_selected.txt ./
COPY planner.py plannerAddData.py streamlit_app.py requirements.txt README.md ./


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Default command to run the app
CMD ["streamlit", "run", "streamlit_app.py"]
