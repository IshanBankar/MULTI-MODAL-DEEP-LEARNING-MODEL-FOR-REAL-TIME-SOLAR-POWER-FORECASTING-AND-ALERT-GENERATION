FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libjpeg62-turbo-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=200 --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]