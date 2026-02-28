FROM python:3.11-slim

WORKDIR /app

# 依存を先に入れてキャッシュ効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリをコピー
COPY app ./app

# Cloud Run は PORT 環境変数（なければ8080）で待ち受け
ENV PORT=8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
