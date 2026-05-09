FROM --platform=linux/amd64 python:3.14-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY app ./app
COPY resources ./resources
RUN PYTHONPATH=/install/lib/python3.14/site-packages python -m app.index

FROM --platform=linux/amd64 python:3.14-slim
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build /app/app ./app
COPY --from=build /app/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=build /app/resources/normalization.json ./resources/normalization.json
COPY --from=build /app/resources/references.faiss ./resources/references.faiss
COPY --from=build /app/resources/references.labels.npy ./resources/references.labels.npy
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--log-level", "warning", "--no-access-log"]
