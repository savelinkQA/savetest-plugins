"""
ReadyAPI Composite Parser Plugin - парсер тест-кейсов ReadyAPI/SoapUI composite project (XML).
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_parser import BaseParser, TestMetadata, ParserError, ParserValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ReadyAPI Composite Parser Plugin", version="1.0.0")

NS = "http://eviware.com/soapui/config"
ELEMENT_ORDER_FILENAME = "element.order"
PROJECT_CONTENT_FILENAME = "project.content"
SUITE_SETTINGS_FILENAME = "settings.xml"


@lru_cache(maxsize=4096)
def _read_suite_id_from_settings_xml(suite_dir: str) -> Optional[str]:
    """
    Извлекает con:testSuite@id из settings.xml в папке сьюта.

    В composite-проектах SoapUI/ReadyAPI settings.xml в директории сьюта имеет вид:
      <con:testSuite id="..."> ... </con:testSuite>
    """
    try:
        d = Path(suite_dir)
        if not d.is_dir():
            return None
        settings_path = d / SUITE_SETTINGS_FILENAME
        if not settings_path.is_file():
            return None
        tree = etree.parse(str(settings_path))
        root = tree.getroot()
        if root is None:
            return None
        # Нужен именно testSuite, иначе пропускаем
        if _local_tag(root) != "testSuite":
            return None
        sid = _attr(root, "id", "").strip()
        return sid or None
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _read_element_order_cached(dir_path: str) -> Optional[tuple]:
    """Кэшированное чтение element.order из dir_path. Возвращает tuple или None."""
    path = Path(dir_path)
    if not path.is_dir():
        return None
    order_file = path / ELEMENT_ORDER_FILENAME
    if not order_file.is_file():
        return None
    try:
        with open(order_file, "r", encoding="utf-8") as f:
            lines = tuple(line.strip() for line in f if line.strip())
        return lines if lines else None
    except Exception:
        return None


def _read_element_order(suite_dir: Path) -> Optional[List[str]]:
    """Читает element.order в папке сьюта. Возвращает список имён файлов (порядок = порядок кейсов в сьюте)."""
    result = _read_element_order_cached(str(suite_dir.resolve()))
    return list(result) if result else None


def _find_project_content_root(start: Path) -> Optional[Path]:
    """Поднимается от start вверх, возвращает директорию, в которой лежит project.content."""
    current = start.resolve() if start.is_dir() else start.parent.resolve()
    while current and current != current.parent:
        if (current / PROJECT_CONTENT_FILENAME).is_file():
            return current
        current = current.parent
    return None


def _read_project_content(project_root: Path) -> Optional[List[str]]:
    """Читает project.content в корне проекта. Возвращает список относительных путей (нормализованных к /)."""
    content_file = project_root / PROJECT_CONTENT_FILENAME
    if not content_file.is_file():
        return None
    try:
        with open(content_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                lines.append(s.replace("\\", "/"))
        return lines
    except Exception:
        return None


def _order_index_in_suite(file_path: Path) -> Optional[int]:
    """Порядковый индекс тест-кейса в сьюте (0-based) по element.order. None если файла нет или element.order отсутствует."""
    order = _read_element_order(file_path.parent)
    if not order:
        return None
    name = file_path.name
    try:
        idx = order.index(name)
        return idx
    except ValueError:
        return None


def _suite_order_index(file_path: Path) -> Optional[int]:
    """
    Порядковый индекс сьюта в проекте (0-based) по element.order в родительской директории.

    Структура composite-проекта:
        project_dir/element.order   ← список дочерних папок в порядке отображения
          suite_dir/element.order   ← список XML-файлов тест-кейсов
            test_case.xml

    Метод ищет имя директории сьюта в element.order проектной директории.
    """
    suite_dir = file_path.parent
    project_dir = suite_dir.parent
    if not project_dir.is_dir():
        return None
    order = _read_element_order_cached(str(project_dir.resolve()))
    if not order:
        return None
    suite_name = suite_dir.name
    try:
        return order.index(suite_name)
    except ValueError:
        return None


def _project_content_order(file_path: Path, project_root: Path) -> Optional[int]:
    """Индекс (0-based) пути к файлу в project.content. None если не найден."""
    content = _read_project_content(project_root)
    if not content:
        return None
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
        rel_str = str(rel).replace("\\", "/")
        return content.index(rel_str)
    except (ValueError, TypeError):
        return None


class ParseRequest(BaseModel):
    file_path: str
    repo_path: Optional[str] = None


class ParseRepoRequest(BaseModel):
    repo_path: str
    file_paths: Optional[List[str]] = None


class CanParseRequest(BaseModel):
    file_path: str


class ParseResponse(BaseModel):
    success: bool
    metadata: List[Dict[str, Any]]
    error: Optional[str] = None


class ParseRepoResponse(BaseModel):
    success: bool
    metadata: List[Dict[str, Any]]
    errors: List[str] = []
    files_processed: int = 0
    files_failed: int = 0


class CanParseResponse(BaseModel):
    can_parse: bool


class HealthResponse(BaseModel):
    status: str
    plugin_name: str
    version: str


def _local_tag(elem) -> str:
    """Локальное имя тега без namespace."""
    if callable(getattr(elem, "tag", None)):
        return ""
    if isinstance(elem.tag, str) and elem.tag.startswith("{"):
        return elem.tag.split("}", 1)[1]
    return elem.tag or ""


def _is_test_case_root(root) -> bool:
    """Проверка, что корень — con:testCase."""
    if root is None:
        return False
    local = _local_tag(root)
    ns = root.nsmap.get(None) or (root.tag.split("}", 1)[0].strip("{}") if isinstance(root.tag, str) and "}" in root.tag else "")
    return local == "testCase" and (ns == NS or root.tag == f"{{{NS}}}testCase")


def _find_children(parent, local_name: str):
    """Дочерние элементы по локальному имени (любой namespace)."""
    if parent is None:
        return []
    return [c for c in parent if _local_tag(c) == local_name]


def _attr(elem, name: str, default: str = "") -> str:
    """Атрибут по локальному имени (без namespace)."""
    if elem is None:
        return default
    for k, v in (elem.attrib or {}).items():
        local = k.split("}", 1)[-1] if "}" in k else k
        if local == name:
            return (v or "").strip() or default
    return default


def _text(elem, default: str = "") -> str:
    """Текстовое содержимое элемента (включая вложенные)."""
    if elem is None:
        return default
    parts = []
    if elem.text:
        parts.append(elem.text)
    for c in elem:
        parts.append(_text(c, ""))
        if c.tail:
            parts.append(c.tail)
    t = "".join(parts).strip()
    return t if t else default


def _first_child(parent, local_name: str):
    """Первый дочерний элемент с заданным локальным именем."""
    for c in parent:
        if _local_tag(c) == local_name:
            return c
    return None


def _extract_assertions(request_elem) -> List[Dict[str, Any]]:
    """Извлекает список assertions из con:request, включая все конфигурационные поля."""
    out = []
    for a in _find_children(request_elem, "assertion"):
        cfg = _first_child(a, "configuration")
        entry = {
            "type": _attr(a, "type"),
            "name": _attr(a, "name") or _attr(a, "type"),
            "id": _attr(a, "id"),
        }
        if cfg is not None:
            codes_el = _first_child(cfg, "codes")
            if codes_el is not None and codes_el.text:
                entry["codes"] = codes_el.text.strip()
            path_el = _first_child(cfg, "path")
            if path_el is not None and path_el.text:
                entry["path"] = path_el.text.strip()
            content_el = _first_child(cfg, "content")
            if content_el is not None and content_el.text:
                entry["content"] = content_el.text.strip()
            for flag in ("allowWildcards", "ignoreNamspaceDifferences", "ignoreComments"):
                el = _first_child(cfg, flag)
                if el is not None and el.text and el.text.strip().lower() == "true":
                    entry[flag] = True
        out.append(entry)
    return out


def _extract_request_step_info(config) -> Dict[str, Any]:
    """Из con:config (RequestStep) извлекает interface, operation, endpoint, полный текст запроса, assertions."""
    info: Dict[str, Any] = {
        "interface": "",
        "operation": "",
        "endpoint": "",
        "request_body": "",
        "assertions": [],
    }
    iface_el = _first_child(config, "interface")
    if iface_el is not None:
        info["interface"] = _text(iface_el)
    op_el = _first_child(config, "operation")
    if op_el is not None:
        info["operation"] = _text(op_el)

    req = _first_child(config, "request")
    if req is None:
        return info

    ep_el = _first_child(req, "endpoint")
    if ep_el is not None:
        info["endpoint"] = _text(ep_el)

    req_body_el = _first_child(req, "request")
    if req_body_el is not None and req_body_el.text:
        info["request_body"] = req_body_el.text.strip()

    info["assertions"] = _extract_assertions(req)
    return info


def _safe_code_fence(content: str, lang: str = "") -> str:
    """
    Возвращает безопасный code-fence для произвольного содержимого.
    Если в content есть ``` — использует 4 бэктика, иначе 3.
    """
    fence = "````" if "```" in content else "```"
    return f"{fence}{lang}\n{content}\n{fence}"


def _inline_or_fence(content: str, lang: str = "xml") -> str:
    """
    Форматирует значение как Markdown:
    - однострочное без бэктиков → inline code
    - однострочное с бэктиком   → inline code двойными бэктиками
    - многострочное             → fenced code block
    """
    if "\n" in content:
        return "\n\n" + _safe_code_fence(content, lang)
    if "`" in content:
        # Double-backtick inline fence (стандарт CommonMark для контента с бэктиком)
        return f" `` {content} ``"
    return f" `{content}`"


def _format_assertions_markdown(assertions: List[Dict[str, Any]]) -> str:
    """
    Форматирует список assertions как Markdown.

    Каждый assertion — отдельный параграф.
    Многострочные ожидаемые значения (XPath content, path) оборачиваются
    в fenced code block, а не в inline-код, так как inline-код не может
    содержать переносы строк.
    """
    if not assertions:
        return ""
    parts = []
    for a in assertions:
        name = a.get("name") or a.get("type", "Assertion")
        atype = a.get("type", "")
        codes = a.get("codes", "")
        path = a.get("path", "")
        content = a.get("content", "")

        section_lines = []

        if codes and ("StatusCode" in atype or "HTTP Status" in atype or "Valid HTTP" in atype):
            section_lines.append(f"**HTTP-статус:** `{codes}`")

        elif "XPath" in atype or "JsonPath" in atype:
            section_lines.append(f"**{name}**")
            if content:
                content_lang = "xml" if content.lstrip().startswith("<") else "text"
                section_lines.append("Ожидается:" + _inline_or_fence(content, content_lang))
            if path:
                path_lang = "xpath" if "XPath" in atype else "jsonpath"
                section_lines.append("Путь:" + _inline_or_fence(path, path_lang))

        elif "Script" in atype:
            section_lines.append(f"**{name}**")
            if content:
                section_lines.append("Скрипт:" + _inline_or_fence(content, "groovy"))

        else:
            section_lines.append(f"**{name}**")

        parts.append("\n".join(section_lines))

    return "\n\n---\n\n".join(parts)


def _format_request_action(step_name: str, request_info: Dict[str, Any]) -> str:
    """Форматирует поле action для request-шага как Markdown с кодовым блоком XML/SOAP."""
    parts = [f"**{step_name}**"]

    iface = request_info.get("interface", "")
    operation = request_info.get("operation", "")
    if iface and operation:
        parts.append(f"Операция: `{iface}.{operation}`")
    elif iface or operation:
        parts.append(f"Операция: `{iface or operation}`")

    endpoint = request_info.get("endpoint", "")
    if endpoint:
        parts.append(f"Endpoint: `{endpoint}`")

    body = request_info.get("request_body", "")
    if body:
        lang = "xml" if body.lstrip().startswith("<") else "text"
        parts.append(f"\n```{lang}\n{body}\n```")

    return "\n".join(parts)


def _format_groovy_action(step_name: str, script: str) -> str:
    """Форматирует поле action для Groovy-шага — полный скрипт в кодовом блоке."""
    if script:
        return f"**{step_name}**\n\n```groovy\n{script}\n```"
    return f"**{step_name}**"


class ReadyAPICompositeParser(BaseParser):
    """Парсер тест-кейсов ReadyAPI/SoapUI composite (XML)."""

    def can_parse(self, file_path: Path) -> bool:
        if file_path.suffix.lower() != ".xml":
            return False
        try:
            with open(file_path, "rb") as f:
                head = f.read(4096)
            if b"testCase" not in head or b"eviware.com/soapui/config" not in head:
                return False
            tree = etree.parse(str(file_path))
            return _is_test_case_root(tree.getroot())
        except Exception:
            return False

    def parse_file(self, file_path: Path, repo_path: Optional[Path] = None) -> List[TestMetadata]:
        try:
            with open(file_path, "rb") as f:
                tree = etree.parse(f)
        except Exception as e:
            raise ParserError(f"Не удалось прочитать XML {file_path}: {e}")

        root = tree.getroot()
        if not _is_test_case_root(root):
            return []

        case_id = _attr(root, "id")
        case_name = _attr(root, "name") or file_path.stem
        file_path_str = str(file_path)

        suite_name = None
        if repo_path is not None and file_path.is_absolute() and str(file_path).startswith(str(repo_path)):
            try:
                rel = file_path.relative_to(repo_path)
                if len(rel.parts) > 1:
                    suite_name = rel.parts[-2]
                else:
                    suite_name = rel.parts[0] if rel.parts else None
            except ValueError:
                pass
        if suite_name is None and file_path.parent and file_path.parent.name:
            suite_name = file_path.parent.name

        # suite_id берём из settings.xml в папке сьюта (fallback для Allure metadata sync)
        suite_id = None
        if file_path.parent and file_path.parent.is_dir():
            suite_id = _read_suite_id_from_settings_xml(str(file_path.parent.resolve()))

        custom_fields = []
        timeout_val = _attr(root, "timeout")
        if timeout_val:
            custom_fields.append({"name": "timeout", "value": timeout_val})
        fail_on_error = _attr(root, "failOnError")
        if fail_on_error:
            custom_fields.append({"name": "failOnError", "value": fail_on_error})

        for settings in _find_children(root, "settings"):
            for s in _find_children(settings, "setting"):
                sid = _attr(s, "id", "")
                if "fileName" in sid and s.text:
                    custom_fields.append({"name": "fileName", "value": (s.text or "").strip()})
                    break

        for props in _find_children(root, "properties"):
            for p in _find_children(props, "property"):
                name_el = _first_child(p, "name")
                value_el = _first_child(p, "value")
                if name_el is not None:
                    custom_fields.append({
                        "name": _text(name_el),
                        "value": _text(value_el) if value_el is not None else "",
                    })

        order_index = _order_index_in_suite(file_path)
        if order_index is not None:
            custom_fields.append({"name": "order_index", "value": str(order_index)})

        suite_order_idx = _suite_order_index(file_path)
        if suite_order_idx is not None:
            custom_fields.append({"name": "suite_order_index", "value": str(suite_order_idx)})

        project_root = _find_project_content_root(file_path.parent)
        if repo_path is not None and repo_path.is_dir():
            repo_resolved = repo_path.resolve()
            if (repo_resolved / PROJECT_CONTENT_FILENAME).is_file():
                project_root = project_root or repo_resolved
        if project_root is not None:
            project_content_order = _project_content_order(file_path, project_root)
            if project_content_order is not None:
                custom_fields.append({"name": "project_content_order", "value": str(project_content_order)})

        steps = []
        seen_step_ids = set()

        for step_el in _find_children(root, "testStep"):
            step_id = _attr(step_el, "id")
            if step_id in seen_step_ids:
                continue
            seen_step_ids.add(step_id)
            step_type = _attr(step_el, "type", "unknown")
            step_name = _attr(step_el, "name", "Step")
            disabled = _attr(step_el, "disabled", "").lower() == "true"

            extra: Dict[str, Any] = {"step_type": step_type, "step_id": step_id}
            if disabled:
                extra["disabled"] = True

            steps.append({
                "action": f"[{step_type}] {step_name}",
                "expected_result": "",
                **extra,
            })

        # breakPoints/testStepId — ссылки на шаги, которые не являются реальными шагами теста.
        # Для отображения в библиотеке не добавляем их в список steps.

        # suite_path — директория сьюта (родитель файла).
        # Бэкенд использует Path(suite_path).parent как Directory в UI,
        # избегая лишнего уровня вложенности (suite dir ≠ Directory, это TestSuite).
        suite_path = str(file_path.parent.resolve())

        metadata = TestMetadata(
            tms=case_id,
            file_path=file_path_str,
            suite_id=suite_id,
            suite_name=suite_name,
            suite_path=suite_path,
            title=case_name,
            description=None,
            function_name=case_name,
            severity=None,
            priority=None,
            tags=[],
            links=[],
            steps=steps,
            iterations=[],
            custom_fields=custom_fields,
        )
        return self._validate_and_process_metadata([metadata], file_path)


parser = ReadyAPICompositeParser()


@app.post("/parse", response_model=ParseResponse)
async def parse_file(request: ParseRequest):
    try:
        file_path = Path(request.file_path)
        if not file_path.exists():
            return ParseResponse(success=False, metadata=[], error=f"Файл не найден: {file_path}")
        repo_path = Path(request.repo_path) if request.repo_path else None
        metadata_list = parser.parse_file(file_path, repo_path=repo_path)
        return ParseResponse(
            success=True,
            metadata=[m.to_dict() for m in metadata_list],
        )
    except (ParserError, ParserValidationError) as e:
        logger.error("Ошибка парсинга: %s", e)
        return ParseResponse(success=False, metadata=[], error=str(e))
    except Exception as e:
        logger.exception("Неожиданная ошибка")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_repo", response_model=ParseRepoResponse)
async def parse_repo(request: ParseRepoRequest):
    """
    Парсит весь репозиторий за один вызов.

    Принимает путь к репозиторию и опциональный список файлов.
    Если file_paths не передан — плагин сам находит все подходящие файлы.

    Дубликаты case_id между файлами фиксируются в поле errors,
    но обработка продолжается (дубликат не включается в metadata).
    Дубликаты внутри одного файла по-прежнему приводят к ошибке файла.
    """
    repo_path = Path(request.repo_path)
    if not repo_path.exists():
        return ParseRepoResponse(
            success=False,
            metadata=[],
            errors=[f"repo_path не найден: {repo_path}"],
        )

    if request.file_paths is not None:
        file_paths = [Path(fp) for fp in request.file_paths]
    else:
        file_paths = []
        for f in repo_path.rglob("*.xml"):
            try:
                rel = str(f.relative_to(repo_path))
                if rel.startswith("docs/") or "/docs/" in rel:
                    continue
                if parser.can_parse(f):
                    file_paths.append(f)
            except Exception:
                pass

    all_metadata: List[Dict[str, Any]] = []
    all_raw: List[Any] = []
    errors: List[str] = []
    files_processed = 0
    files_failed = 0

    for file_path in file_paths:
        try:
            if not file_path.exists():
                errors.append(f"Файл не найден: {file_path}")
                files_failed += 1
                continue
            metadata_list = parser.parse_file(file_path, repo_path=repo_path)
            all_raw.extend(metadata_list)
            files_processed += 1
        except (ParserError, ParserValidationError) as e:
            errors.append(f"{file_path}: {e}")
            files_failed += 1
        except Exception as e:
            logger.error("Неожиданная ошибка при parse_repo для %s: %s", file_path, e)
            errors.append(f"{file_path}: неожиданная ошибка — {e}")
            files_failed += 1

    clean, dup_errors = parser.check_cross_file_duplicates(all_raw)
    errors.extend(dup_errors)
    all_metadata = [m.to_dict() for m in clean]

    return ParseRepoResponse(
        success=files_failed == 0 and not dup_errors,
        metadata=all_metadata,
        errors=errors,
        files_processed=files_processed,
        files_failed=files_failed,
    )


@app.post("/can_parse", response_model=CanParseResponse)
async def can_parse_file(request: CanParseRequest):
    try:
        file_path = Path(request.file_path)
        return CanParseResponse(can_parse=parser.can_parse(file_path))
    except Exception as e:
        logger.error("Ошибка can_parse: %s", e)
        return CanParseResponse(can_parse=False)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        plugin_name="readyapi-composite-parser",
        version="1.0.0",
    )


@app.get("/config")
async def get_config():
    try:
        plugin_json_path = Path(__file__).parent.parent / "plugin.json"
        with open(plugin_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Ошибка чтения конфигурации: %s", e)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить конфигурацию: {e}")


@app.get("/")
async def root():
    return {
        "plugin": "readyapi-composite-parser",
        "version": "1.0.0",
        "endpoints": ["/parse", "/can_parse", "/health", "/config"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
