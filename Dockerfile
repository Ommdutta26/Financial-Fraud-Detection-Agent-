# ==============================
# Base Image
# ==============================
FROM python:3.11-slim

# ==============================
# Set Environment Variables
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# Set Working Directory
# ==============================
WORKDIR /app

# ==============================
# Install System Dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Copy Requirements First (cache optimization)
# ==============================
COPY requirements.txt .

# ==============================
# Install Dependencies
# ==============================
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# Copy Project Files
# ==============================
COPY . .

# ==============================
# Expose Streamlit Port
# ==============================
EXPOSE 8501

# ==============================
# Run Streamlit App
# ==============================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]