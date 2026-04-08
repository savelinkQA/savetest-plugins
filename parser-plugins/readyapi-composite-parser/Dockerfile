FROM python:3.11-slim

WORKDIR /app

COPY base_parser.py /app/
COPY plugin.json /app/
COPY src/ /app/src/
COPY requirements.txt /app/

# PyPI по умолчанию даёт ~15 с на операцию; крупные wheel (lxml) на медленной сети не успевают.
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

CMD ["uvicorn", "src.parser:app", "--host", "0.0.0.0", "--port", "8000"]
