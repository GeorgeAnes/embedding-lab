FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lab lab
COPY data/corpus.jsonl data/queries.json data/queries.sha256 data/NOTICE.md data/
COPY web web
# Results are written to /app/results; mount a folder there to keep them.
CMD ["python", "-m", "lab", "eval"]
