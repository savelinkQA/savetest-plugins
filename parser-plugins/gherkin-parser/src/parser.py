"""
Gherkin Parser Plugin - FastAPI сервис для парсинга Gherkin файлов
"""

import json
import re
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Добавляем родительскую директорию в путь для импорта base_parser
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_parser import BaseParser, TestMetadata, ParserError, ParserValidationError

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gherkin Parser Plugin", version="1.0.0")


class ParseRequest(BaseModel):
    """Запрос на парсинг файла"""
    file_path: str
    repo_path: Optional[str] = None


class ParseRepoRequest(BaseModel):
    """Запрос на парсинг всего репозитория"""
    repo_path: str
    file_paths: Optional[List[str]] = None


class CanParseRequest(BaseModel):
    """Запрос на проверку возможности парсинга"""
    file_path: str


class ParseResponse(BaseModel):
    """Ответ с результатами парсинга"""
    success: bool
    metadata: List[Dict[str, Any]]
    error: Optional[str] = None


class ParseRepoResponse(BaseModel):
    """Ответ на парсинг всего репозитория"""
    success: bool
    metadata: List[Dict[str, Any]]
    errors: List[str] = []
    files_processed: int = 0
    files_failed: int = 0


class CanParseResponse(BaseModel):
    """Ответ на проверку возможности парсинга"""
    can_parse: bool


class HealthResponse(BaseModel):
    """Статус плагина"""
    status: str
    plugin_name: str
    version: str


class GherkinParser(BaseParser):
    """Парсер для Gherkin файлов"""
    
    TAG_PATTERNS = {
        'tms': re.compile(r'@(?:tms|case):([^\s]+)'),
        'suite': re.compile(r'@suite:([^\s]+)'),
        'severity': re.compile(r'@severity:([^\s]+)'),
        'tag': re.compile(r'@tag:([^\s]+)'),
        'issue': re.compile(r'@issue:([^\s]+)'),
        'link': re.compile(r'@link:([^\s]+)'),
    }
    
    def can_parse(self, file_path: Path) -> bool:
        """Проверяет, является ли файл Gherkin файлом"""
        return file_path.suffix == '.feature'
    
    def parse_file(self, file_path: Path) -> List[TestMetadata]:
        """Парсит Gherkin файл и извлекает метаданные."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ParserError(f"Не удалось прочитать файл {file_path}: {e}")
        
        try:
            metadata_list = self._parse_feature(content, str(file_path))
        except Exception as e:
            raise ParserError(f"Ошибка парсинга Gherkin файла {file_path}: {e}")
        
        return self._validate_and_process_metadata(metadata_list, file_path)
    
    def _parse_feature(self, content: str, file_path: str) -> List[TestMetadata]:
        """Парсит содержимое .feature файла"""
        metadata_list = []
        lines = content.split('\n')
        
        # Читаем suite-level savetest-комментарии из начала файла
        suite_meta = self._parse_savetest_comments(lines)
        
        feature_suite_id = None
        feature_suite_name = None
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Feature:'):
                feature_suite_name = stripped.split(':', 1)[1].strip() if ':' in stripped else 'Feature'
                break
            elif stripped.startswith('@suite:'):
                match = self.TAG_PATTERNS['suite'].search(stripped)
                if match:
                    feature_suite_id = match.group(1)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith(('Scenario:', 'Scenario Outline:')):
                tags = []
                case_comment_lines = []
                j = i - 1
                while j >= 0:
                    prev_line = lines[j].strip()
                    if prev_line.startswith('@'):
                        tags.insert(0, prev_line)
                        j -= 1
                    elif prev_line == '':
                        j -= 1
                    elif prev_line.startswith('#'):
                        # Собираем savetest_case_* комментарии
                        case_comment_lines.insert(0, prev_line)
                        j -= 1
                    else:
                        break
                
                scenario_name = line.split(':', 1)[1].strip() if ':' in line else ''
                
                # Извлекаем case-level метаданные из собранных комментариев
                case_meta = self._extract_case_savetest_comments(case_comment_lines)
                case_custom_fields = self._extract_case_custom_fields(case_comment_lines)
                
                metadata = self._extract_from_tags(
                    tags, feature_suite_id, feature_suite_name,
                    scenario_name, file_path, suite_meta, case_meta, case_custom_fields
                )
                
                if metadata:
                    steps_list = []
                    k = i + 1
                    examples_start = None
                    
                    while k < len(lines):
                        step_line = lines[k].strip()
                        if step_line.startswith(('Given', 'When', 'Then', 'And', 'But')):
                            steps_list.append({
                                'action': step_line,
                                'expected_result': ''
                            })
                            k += 1
                        elif step_line == '' or step_line.startswith('#'):
                            k += 1
                        elif step_line.startswith('|'):
                            k += 1
                        elif step_line.startswith(('Examples:', 'Example:')):
                            examples_start = k
                            k += 1
                            # Пропускаем блок Examples (таблица до первой не-| строки), чтобы собрать шаги после него
                            while k < len(lines):
                                next_line = lines[k].strip()
                                if next_line.startswith('|') or next_line == '' or next_line.startswith('#'):
                                    k += 1
                                else:
                                    break
                        elif step_line.startswith(('Scenarios:', 'Scenario:')):
                            break
                        elif step_line.startswith('@'):
                            break
                        elif step_line.startswith(('Feature:', 'Background:', 'Rule:')):
                            break
                        else:
                            k += 1
                    
                    metadata.steps = steps_list
                    
                    if examples_start is not None:
                        iterations = self._extract_iterations_from_examples(lines, examples_start)
                        metadata.iterations = iterations
                    
                    metadata_list.append(metadata)
            
            i += 1
        
        return metadata_list
    
    def _extract_from_tags(
        self,
        tags: List[str],
        feature_suite_id: Optional[str],
        feature_suite_name: Optional[str],
        scenario_name: str,
        file_path: str,
        suite_meta: Optional[Dict[str, str]] = None,
        case_meta: Optional[Dict[str, str]] = None,
        case_custom_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[TestMetadata]:
        """Извлекает метаданные из списка тегов и savetest-комментариев"""
        all_tags_text = ' '.join(tags)
        
        tms_match = self.TAG_PATTERNS['tms'].search(all_tags_text)
        if not tms_match:
            return None
        
        tms_value = tms_match.group(1)
        
        suite_id = feature_suite_id
        suite_match = self.TAG_PATTERNS['suite'].search(all_tags_text)
        if suite_match:
            suite_id = suite_match.group(1)
        
        suite_name = feature_suite_name
        
        severity_value = None
        priority_value = None
        severity_match = self.TAG_PATTERNS['severity'].search(all_tags_text)
        if severity_match:
            severity = severity_match.group(1)
            severity_value = severity
            priority_value = self._map_severity_to_priority(severity)
        
        tag_matches = self.TAG_PATTERNS['tag'].findall(all_tags_text)
        tags_list = list(dict.fromkeys(tag_matches)) if tag_matches else []
        
        links_list = []
        issue_matches = self.TAG_PATTERNS['issue'].findall(all_tags_text)
        if issue_matches:
            for issue in issue_matches:
                links_list.append({
                    'name': issue,
                    'value': issue
                })
        
        link_matches = self.TAG_PATTERNS['link'].findall(all_tags_text)
        if link_matches:
            for link in link_matches:
                links_list.append({
                    'name': 'link',
                    'value': link
                })
        
        suite_meta = suite_meta or {}
        case_meta = case_meta or {}
        case_custom_fields = case_custom_fields or []
        
        return TestMetadata(
            tms=tms_value,
            file_path=file_path,
            suite_id=suite_id,
            suite_name=suite_name,
            title=scenario_name,
            description=case_meta.get('description'),
            function_name=scenario_name,
            severity=severity_value,
            priority=priority_value,
            tags=tags_list,
            links=links_list,
            suite_status=suite_meta.get('status'),
            suite_description=suite_meta.get('description'),
            suite_author=suite_meta.get('author'),
            suite_created_at=suite_meta.get('created_at'),
            estimated_time=case_meta.get('estimated_time'),
            custom_fields=case_custom_fields,
        )
    
    def _extract_iterations_from_examples(self, lines: List[str], examples_start: int) -> List[Dict[str, Any]]:
        """Извлекает iterations из секции Examples для Scenario Outline"""
        iterations = []
        
        try:
            k = examples_start + 1
            
            while k < len(lines):
                line = lines[k].strip()
                if line.startswith('|'):
                    break
                elif line == '' or line.startswith('#'):
                    k += 1
                else:
                    return iterations
            
            if k >= len(lines):
                return iterations
            
            header_line = lines[k].strip()
            if not header_line.startswith('|'):
                return iterations
            
            param_names = [col.strip() for col in header_line.split('|')[1:-1] if col.strip()]
            
            if not param_names:
                return iterations
            
            k += 1
            iteration_idx = 1
            
            while k < len(lines):
                line = lines[k].strip()
                
                if line.startswith('|'):
                    values = [col.strip() for col in line.split('|')[1:-1] if col.strip()]
                    
                    if values and len(values) == len(param_names):
                        iteration = {
                            'iteration_id': str(iteration_idx),
                            'params': [
                                {'name': param_names[i], 'value': values[i]}
                                for i in range(len(param_names))
                            ]
                        }
                        iterations.append(iteration)
                        iteration_idx += 1
                    
                    k += 1
                elif line == '' or line.startswith('#'):
                    k += 1
                else:
                    break
        
        except Exception:
            pass
        
        return iterations


# Создаем единственный экземпляр парсера
parser = GherkinParser()


@app.post("/parse", response_model=ParseResponse)
async def parse_file(request: ParseRequest):
    """Парсит файл и возвращает метаданные"""
    try:
        file_path = Path(request.file_path)
        metadata_list = parser.parse_file(file_path)
        
        return ParseResponse(
            success=True,
            metadata=[m.to_dict() for m in metadata_list]
        )
    
    except (ParserError, ParserValidationError) as e:
        logger.error(f"Ошибка парсинга: {e}")
        return ParseResponse(
            success=False,
            metadata=[],
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_repo", response_model=ParseRepoResponse)
async def parse_repo(request: ParseRepoRequest):
    """
    Парсит весь репозиторий за один вызов.

    Принимает путь к репозиторию и опциональный список файлов.
    Если file_paths не передан — плагин сам находит все .feature файлы.

    Дубликаты case_id между файлами фиксируются в поле errors,
    но обработка продолжается (дубликат не включается в metadata).
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
        for f in repo_path.rglob("*.feature"):
            try:
                rel = str(f.relative_to(repo_path))
                if rel.startswith("docs/") or "/docs/" in rel:
                    continue
                file_paths.append(f)
            except Exception:
                pass
        file_paths = sorted(file_paths)

    all_raw = []
    errors: List[str] = []
    files_processed = 0
    files_failed = 0

    for file_path in file_paths:
        try:
            if not file_path.exists():
                errors.append(f"Файл не найден: {file_path}")
                files_failed += 1
                continue
            metadata_list = parser.parse_file(file_path)
            all_raw.extend(metadata_list)
            files_processed += 1
        except (ParserError, ParserValidationError) as e:
            errors.append(f"{file_path}: {e}")
            files_failed += 1
        except Exception as e:
            logger.error(f"Неожиданная ошибка при parse_repo для {file_path}: {e}")
            errors.append(f"{file_path}: неожиданная ошибка — {e}")
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


@app.post("/can_parse", response_model=CanParseResponse)
async def can_parse_file(request: CanParseRequest):
    """Проверяет, может ли парсер обработать файл"""
    try:
        file_path = Path(request.file_path)
        can_parse = parser.can_parse(file_path)
        return CanParseResponse(can_parse=can_parse)
    except Exception as e:
        logger.error(f"Ошибка проверки: {e}")
        return CanParseResponse(can_parse=False)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка работоспособности плагина"""
    return HealthResponse(
        status="ok",
        plugin_name="gherkin-parser",
        version="1.0.0"
    )


@app.get("/config")
async def get_config():
    """Возвращает конфигурацию плагина из plugin.json"""
    try:
        plugin_json_path = Path(__file__).parent.parent / "plugin.json"
        with open(plugin_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Ошибка чтения конфигурации: {e}")
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить конфигурацию: {e}")


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "plugin": "gherkin-parser",
        "version": "1.0.0",
        "endpoints": ["/parse", "/can_parse", "/health", "/config"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

