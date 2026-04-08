# ReadyAPI Composite Parser

Плагин парсера тест-кейсов для проектов **ReadyAPI / SoapUI** в формате **composite project** (каждый тест-кейс в отдельном XML-файле).

## Поддерживаемый формат

- Корневой элемент файла: `con:testCase` (namespace `http://eviware.com/soapui/config`).
- Один XML-файл = один тест-кейс (один элемент метаданных).
- Файлы с корнем `con:operation`, `con:soapui-project`, `con:testSuite` и т.п. не считаются тестами — для них парсер возвращает пустой список.

## Что извлекается

- **tms** — GUID тест-кейса (`con:testCase@id`).
- **title** — имя кейса (`con:testCase@name`) или имя файла.
- **suite_name** — имя родительской папки (тест-сьют).
- **steps** — для каждого шага:
  - **action** — имя шага (`con:testStep@name`).
  - **expected_result** — краткое описание (типы assertions для request, "Groovy script" для groovy, "Delay" для delay).
  - **step_type** — `groovy`, `request`, `delay` или `reference`.
  - **step_id** — GUID шага.
  - Для **groovy**: `script_preview` (первые 400 символов скрипта).
  - Для **request**: `endpoint`, `request_preview` (обрезок тела), `assertions` — список `{type, name, path, content, codes}`.
  - Для **reference** (con:testStepId): `test_step_id` — GUID ссылки на шаг в другом файле.
- **custom_fields** — timeout, failOnError, fileName из settings, свойства кейса (con:properties), а также:
  - **order_index** — порядковый индекс кейса внутри сьюта (0-based), из `element.order` в папке сьюта;
  - **project_content_order** — глобальный порядок в проекте (0-based), из `project.content` в корне проекта.

## Файлы порядка (element.order, project.content)

Парсер при наличии файлов читает их и добавляет в метаданные порядок:

- **element.order** — файл в папке сьюта (рядом с XML тест-кейсами), по одному имени файла на строку. Порядок строк задаёт порядок кейсов в сьюте. В `custom_fields` записывается **order_index** (0-based индекс текущего файла в этом списке).
- **project.content** — файл в корне ReadyAPI-проекта, строки вида `SuiteFolder\filename.xml`. Задаёт глобальный порядок элементов проекта. Корень ищется по пути к файлу (вверх по дереву) или по переданному `repo_path`. В `custom_fields` записывается **project_content_order** (0-based индекс относительного пути к текущему файлу в этом списке).

Если файла нет или текущий файл не найден в списке, соответствующее поле не добавляется.

## Ограничения

- Ссылки на шаги (`con:testStepId`) не разрешаются — извлекается только ID ссылки, без загрузки содержимого шага из других файлов.
- Итерации (параметризация) в формате composite не извлекаются — поле `iterations` всегда пустое.

## API

Стандартные endpoints плагина: `POST /parse`, `POST /can_parse`, `GET /health`, `GET /config`.

В запросе `/parse` можно передать опционально `repo_path` — тогда `suite_name` вычисляется относительно этой корневой директории.

## Зависимости

- Python 3.11+
- fastapi, uvicorn, pydantic, lxml
