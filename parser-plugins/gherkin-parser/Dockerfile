FROM python:3.11-slim

WORKDIR /app

# Копируем базовые классы
COPY base_parser.py /app/
COPY plugin.json /app/

# Копируем код плагина
COPY src/ /app/src/
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем API сервер
CMD ["uvicorn", "src.parser:app", "--host", "0.0.0.0", "--port", "8000"]

