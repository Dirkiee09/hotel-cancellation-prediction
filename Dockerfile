FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e . -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
