# Parser Plugins

Плагинная система для парсеров тестовых файлов.

## Структура

Каждый плагин представляет собой отдельный микросервис с FastAPI, который предоставляет REST API для парсинга файлов.

### Текущие плагины

1. **python-parser** - Парсер для Python тестов (pytest, unittest, nose)
2. **gherkin-parser** - Парсер для Gherkin файлов (Cucumber, Behave)

## Формат плагина

Каждый плагин должен содержать следующие файлы:

```
plugin-name/
├── plugin.json          # Конфигурация плагина
├── Dockerfile           # Docker образ плагина
├── requirements.txt     # Python зависимости
├── base_parser.py       # Базовые классы для парсинга
├── src/
│   └── parser.py        # Основной код парсера с FastAPI
└── README.md            # Документация плагина
```

### plugin.json

Обязательный конфигурационный файл с метаданными плагина:

```json
{
  "name": "plugin-name",
  "version": "1.0.0",
  "display_name": "Display Name",
  "description": "Description of what this parser does",
  "language": "language-identifier",
  "supported_extensions": [".ext"],
  "file_patterns": ["**/*_test.ext"],
  "api_version": "1.0",
  "endpoints": {
    "parse": "/parse",
    "can_parse": "/can_parse",
    "health": "/health",
    "config": "/config"
  },
  "config": {
    "port": 8000,
    "timeout": 30
  }
}
```

### Обязательные API endpoints

Каждый плагин должен реализовать следующие endpoints:

#### POST /parse

Парсит файл и возвращает метаданные.

**Request:**
```json
{
  "file_path": "/path/to/test/file.py",
  "repo_path": "/path/to/repository"
}
```

**Response:**
```json
{
  "success": true,
  "metadata": [
    {
      "tms": "case-guid",
      "file_path": "/path/to/file",
      "suite_id": "suite-guid",
      "title": "Test title",
      "description": "Test description",
      ...
    }
  ],
  "error": null
}
```

#### POST /can_parse

Проверяет, может ли парсер обработать данный файл.

**Request:**
```json
{
  "file_path": "/path/to/file.py"
}
```

**Response:**
```json
{
  "can_parse": true
}
```

#### GET /health

Проверка работоспособности плагина.

**Response:**
```json
{
  "status": "ok",
  "plugin_name": "plugin-name",
  "version": "1.0.0"
}
```

#### GET /config

Возвращает конфигурацию плагина из `plugin.json`. Используется бекендом для автоматического обнаружения и регистрации плагина.

**Response:**
```json
{
  "name": "plugin-name",
  "version": "1.0.0",
  "display_name": "Display Name",
  "description": "Description",
  "language": "language-identifier",
  "supported_extensions": [".ext"],
  "file_patterns": ["**/*_test.ext"],
  "api_version": "1.0",
  "endpoints": {
    "parse": "/parse",
    "can_parse": "/can_parse",
    "health": "/health",
    "config": "/config"
  },
  "config": {
    "port": 8000,
    "timeout": 30
  }
}
```

## Разработка нового плагина

### 1. Создайте структуру директорий

```bash
mkdir -p parser-plugins/my-parser/src
cd parser-plugins/my-parser
```

### 2. Создайте plugin.json

Скопируйте и адаптируйте конфигурацию из существующих плагинов.

### 3. Скопируйте base_parser.py

```bash
cp ../python-parser/base_parser.py .
```

### 4. Реализуйте парсер

Создайте `src/parser.py` с FastAPI приложением и реализацией парсера.

Ваш парсер должен наследоваться от `BaseParser` и реализовывать методы:
- `can_parse(file_path: Path) -> bool`
- `parse_file(file_path: Path) -> List[TestMetadata]`

### 5. Создайте Dockerfile

Используйте следующий шаблон:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY base_parser.py /app/
COPY plugin.json /app/
COPY src/ /app/src/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.parser:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6. Создайте requirements.txt

Минимальные зависимости:
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
```

Добавьте специфичные зависимости для вашего парсера.

### 7. Добавьте плагин в docker-compose.yml

```yaml
my-parser-plugin:
  build:
    context: ./parser-plugins/my-parser
    dockerfile: Dockerfile
  container_name: my-parser-plugin
  restart: unless-stopped
  volumes:
    - git_repos:/app/git_repos:ro
  networks:
    - savetest-network
  environment:
    - PLUGIN_NAME=my-parser
    - LOG_LEVEL=INFO
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 10s
    timeout: 5s
```

### 8. Зарегистрируйте плагин в бекенде

Бекенд автоматически обнаруживает плагины через переменные окружения. Добавьте URL вашего плагина в переменную `PLUGIN_URLS` в `docker-compose.yml`:

```yaml
backend:
  environment:
    # Список URL плагинов через запятую
    - PLUGIN_URLS=http://python-parser-plugin:8000,http://gherkin-parser-plugin:8000,http://my-parser-plugin:8000
```

**Важно:** Бекенд автоматически получает конфигурацию плагина через `/config` endpoint при старте. Не нужно вручную регистрировать плагин в коде!

**Альтернативный формат:**
```yaml
backend:
  environment:
    - PLUGIN_1_URL=http://python-parser-plugin:8000
    - PLUGIN_2_URL=http://gherkin-parser-plugin:8000
    - PLUGIN_3_URL=http://my-parser-plugin:8000
```
```
    retries: 3
    start_period: 10s
```

### 9. Документация

Создайте README.md для вашего плагина с описанием:
- Поддерживаемые фреймворки
- Поддерживаемые декораторы/аннотации
- Примеры использования
- Особенности парсинга
