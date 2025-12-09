# AiLowCurrentEngineerPy/Dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python-headless pytesseract


COPY app /app/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
