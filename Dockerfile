# ---------------------------------------------------------------
# Physical AI & Humanoid Robotics Book Platform — Backend Image
# Target: Hugging Face Spaces (Docker SDK)
# Port:   7860  (HF Spaces requirement)
# ---------------------------------------------------------------

FROM python:3.11-slim

# --- system dependencies -----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# --- create persistent-storage directory -------------------------
# HF Spaces mounts a persistent volume at /data.
# SQLite database will live at /data/app.db
RUN mkdir -p /data

# --- working tree layout -----------------------------------------
# /app/ai/       <- project/ai/   (imported via sys.path /app)
# /app/backend/  <- project/backend/  (uvicorn entry point)
WORKDIR /app

# Copy the shared AI module first (changes less often)
COPY project/ai/ ./ai/

# Copy the FastAPI backend
COPY project/backend/ ./backend/

# --- install Python dependencies ----------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# --- runtime environment -----------------------------------------
# PYTHONPATH=/app  →  `import ai` resolves to /app/ai/
# PORT=7860        →  HF Spaces expects traffic on 7860
ENV PYTHONPATH=/app \
    PORT=7860

# The remaining secrets (API keys, DATABASE_URL, etc.) are injected
# at runtime by HF Spaces via the Space Settings → Repository secrets
# UI — never baked into the image.

# Switch into the backend directory so uvicorn finds main.py
WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
