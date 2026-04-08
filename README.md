# Parser Plugins

Плагинная система для парсеров тестовых файлов.

## Структура

Каждый плагин представляет собой отдельный микросервис с FastAPI, который предоставляет REST API для парсинга файлов.

### Текущие плагины

| Плагин | Язык/формат | Фреймворки |
|---|---|---|
| **python-parser** | Python `.py` | pytest, unittest |
| **gherkin-parser** | Gherkin `.feature` | Cucumber, Behave |

## Режимы парсинга

Плагины поддерживают два режима работы:

### `/parse` — парсинг одного файла (legacy)

Обрабатывает один файл за вызов. Используется как fallback для совместимости.

### `/parse_repo` — парсинг всего репозитория (рекомендуется)

Обрабатывает весь репозиторий или список файлов за **один HTTP-вызов**.

**Преимущества:**
- Значительно меньше HTTP-запросов (1 на язык вместо N на каждый файл)
- Плагин может использовать внутренний кэш в рамках всего репозитория (особенно эффективно для форматов с общими конфигурационными файлами)
- Дедупликация `case_id` внутри одного языка выполняется на стороне плагина

Бэкенд автоматически выбирает `/parse_repo` если плагин его поддерживает (endpoint указан в `plugin.json`). Для плагинов без `/parse_repo` применяется per-file режим.

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
  "version": "1.1.0",
  "display_name": "Display Name",
  "description": "Description of what this parser does",
  "language": "language-identifier",
  "supported_extensions": [".ext"],
  "file_patterns": ["**/*_test.ext"],
  "api_version": "1.1",
  "endpoints": {
    "parse": "/parse",
    "parse_repo": "/parse_repo",
    "can_parse": "/can_parse",
    "health": "/health",
    "config": "/config"
  },
  "config": {
    "port": 8000,
    "timeout": 30,
    "repo_timeout": 300
  }
}
```

**Поля `config`:**

| Поле | По умолчанию | Описание |
|---|---|---|
| `timeout` | `30` | Таймаут (сек) для `/parse` — один файл |
| `repo_timeout` | `300` | Таймаут (сек) для `/parse_repo` — весь репозиторий |

### Обязательные API endpoints

#### POST /parse

Парсит один файл и возвращает метаданные.

**Request:**
```json
{
  "file_path": "/path/to/test/file.py",
  "repo_path": "/path/to/repository"
}
```

> `repo_path` — опциональный. Передаётся для корректного определения `suite_name` и порядка файлов.

**Response:**
```json
{
  "success": true,
  "metadata": [
    {
      "tms": "case-guid",
      "file_path": "/path/to/file",
      "suite_id": "suite-guid",
      "suite_name": "Suite Name",
      "title": "Test title",
      "description": "Test description",
      "severity": "normal",
      "priority": "normal",
      "tags": [],
      "links": [],
      "steps": [],
      "iterations": [],
      "custom_fields": []
    }
  ],
  "error": null
}
```

#### POST /parse_repo

Парсит весь репозиторий за один вызов.

**Request:**
```json
{
  "repo_path": "/path/to/repository",
  "file_paths": ["/path/to/file1.py", "/path/to/file2.py"]
}
```

> `file_paths` — опциональный список файлов. Если не передан, плагин сам обходит репозиторий по своим паттернам из `plugin.json`.

**Response:**
```json
{
  "success": true,
  "metadata": [
    {
      "tms": "case-guid",
      "file_path": "/path/to/file",
      "suite_id": "suite-guid",
      "suite_name": "Suite Name",
      "title": "Test title",
      "steps": [],
      "custom_fields": []
    }
  ],
  "errors": [],
  "files_processed": 42,
  "files_failed": 0
}
```

**Логика обработки ошибок в `/parse_repo`:**
- Ошибка парсинга одного файла → файл попадает в `errors`, обработка продолжается
- Дубликат `case_id` между файлами → попадает в `errors`, второй экземпляр не включается в `metadata`
- Дубликат `case_id` внутри файла → файл попадает в `errors` (критическая ошибка файла)
- `success = false` только если есть `files_failed > 0` или дубликаты `case_id` между файлами

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

**Response:**
```json
{
  "status": "ok",
  "plugin_name": "plugin-name",
  "version": "1.1.0"
}
```

#### GET /config

Возвращает конфигурацию плагина из `plugin.json`. Используется бэкендом для автоматического обнаружения и регистрации плагина при старте.

## Разработка нового плагина

### 1. Создайте структуру директорий

```bash
mkdir -p parser-plugins/my-parser/src
cd parser-plugins/my-parser
```

### 2. Скопируйте base_parser.py

```bash
cp ../python-parser/base_parser.py .
```

`base_parser.py` содержит базовые классы (`BaseParser`, `TestMetadata`) и метод `check_cross_file_duplicates` для дедупликации в `/parse_repo`.

### 3. Создайте plugin.json

Скопируйте и адаптируйте конфигурацию из существующих плагинов. Обязательно добавьте `"parse_repo": "/parse_repo"` в секцию `endpoints`.

### 4. Реализуйте парсер

Создайте `src/parser.py` с FastAPI приложением.

Ваш парсер должен наследоваться от `BaseParser` и реализовывать:
- `can_parse(file_path: Path) -> bool`
- `parse_file(file_path: Path) -> List[TestMetadata]`

Добавьте эндпоинты `/parse` и `/parse_repo`:

```python
from base_parser import BaseParser, TestMetadata, ParserError, ParserValidationError

class ParseRepoRequest(BaseModel):
    repo_path: str
    file_paths: Optional[List[str]] = None

class ParseRepoResponse(BaseModel):
    success: bool
    metadata: List[Dict[str, Any]]
    errors: List[str] = []
    files_processed: int = 0
    files_failed: int = 0

@app.post("/parse_repo", response_model=ParseRepoResponse)
async def parse_repo(request: ParseRepoRequest):
    repo_path = Path(request.repo_path)
    if not repo_path.exists():
        return ParseRepoResponse(success=False, metadata=[],
                                 errors=[f"repo_path не найден: {repo_path}"])

    if request.file_paths is not None:
        file_paths = [Path(fp) for fp in request.file_paths]
    else:
        # Обход репозитория по вашим паттернам
        file_paths = [...]

    all_raw, errors = [], []
    files_processed = files_failed = 0

    for file_path in file_paths:
        try:
            all_raw.extend(parser.parse_file(file_path))
            files_processed += 1
        except (ParserError, ParserValidationError) as e:
            errors.append(f"{file_path}: {e}")
            files_failed += 1

    clean, dup_errors = parser.check_cross_file_duplicates(all_raw)
    errors.extend(dup_errors)

    return ParseRepoResponse(
        success=files_failed == 0 and not dup_errors,
        metadata=[m.to_dict() for m in clean],
        errors=errors,
        files_processed=files_processed,
        files_failed=files_failed,
    )
```

### 5. Создайте Dockerfile

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
    retries: 3
    start_period: 10s
```

### 8. Зарегистрируйте плагин в бэкенде

Бэкенд автоматически обнаруживает плагины через переменные окружения. Добавьте URL вашего плагина в переменную `PLUGIN_URLS`:

```yaml
backend:
  environment:
    - PLUGIN_URLS=http://python-parser-plugin:8000,http://gherkin-parser-plugin:8000,http://my-parser-plugin:8000
```

**Альтернативный формат:**
```yaml
backend:
  environment:
    - PLUGIN_1_URL=http://python-parser-plugin:8000
    - PLUGIN_2_URL=http://gherkin-parser-plugin:8000
    - PLUGIN_3_URL=http://my-parser-plugin:8000
```

Бэкенд автоматически получает конфигурацию плагина через `/config` endpoint при старте. Вручную регистрировать в коде не нужно.

### 9. Документация

Создайте `README.md` для вашего плагина с описанием:
- Поддерживаемые фреймворки
- Поддерживаемые декораторы/аннотации
- Примеры использования
- Особенности парсинга

## Проверки дубликатов

Система обеспечивает трёхуровневую защиту от дубликатов:

| Уровень | Где проверяется | Что проверяется |
|---|---|---|
| Внутри файла | Плагин (`_validate_and_process_metadata`) | Дубликаты `case_id` в одном файле |
| Между файлами (один язык) | Плагин (`check_cross_file_duplicates` в `/parse_repo`) | Дубликаты `case_id` между файлами одного языка |
| Между языками | Экстрактор (`_extract_via_parse_repo`) | Дубликаты `case_id` при слиянии результатов разных плагинов |
| Suite/директории | Экстрактор (`_validate_extracted_metadata`) | `suite_id` в разных директориях |
