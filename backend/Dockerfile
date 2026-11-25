FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY quantmind ./quantmind
ENV QM_STATE_DIR=/data QM_AUTONOMY_LEVEL=3
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "quantmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
