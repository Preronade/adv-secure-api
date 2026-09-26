FROM python:3.12-slim
WORKDIR /app
RUN useradd -m appuser
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src/ src/
USER appuser
EXPOSE 8000
CMD ["uvicorn", "--app-dir", "src", "api:app", "--host", "0.0.0.0", "--port", "8000"]