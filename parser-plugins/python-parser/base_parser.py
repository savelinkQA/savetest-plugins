"""
Базовые классы и типы для парсеров метаданных тестов.
Копия из save-test-backend/app/utils/parsers/base.py для использования в плагине.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Базовое исключение для ошибок парсинга"""
    pass


class ParserValidationError(ParserError):
    """Исключение для ошибок валидации метаданных"""
    pass


class AllureSeverity(str, Enum):
    """Уровни важности Allure"""
    BLOCKER = "blocker"
    CRITICAL = "critical"
    NORMAL = "normal"
    MINOR = "minor"
    TRIVIAL = "trivial"


@dataclass
class TestMetadata:
    """Метаданные теста"""
    # Обязательные поля
    tms: str  # case_id (GUID)
    file_path: str
    
    # Suite информация
    suite_id: Optional[str] = None  # GUID test suite
    suite_name: Optional[str] = None
    
    # Дополнительные поля
    title: Optional[str] = None
    description: Optional[str] = None
    function_name: Optional[str] = None
    
    # Приоритет и теги
    severity: Optional[str] = None
    priority: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    
    # Шаги выполнения (извлечённые из allure.step)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Итерации (параметризация теста)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Suite-level метаданные из savetest-комментариев (# savetest_*)
    suite_status: Optional[str] = None
    suite_description: Optional[str] = None
    suite_author: Optional[str] = None
    suite_created_at: Optional[str] = None
    
    # Case-level метаданные из savetest-комментариев (# savetest_case_*)
    estimated_time: Optional[str] = None
    custom_fields: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self) -> None:
        """
        Валидация метаданных теста.
        
        Raises:
            ParserValidationError: если метаданные невалидны
        """
        errors = []
        
        # Проверка обязательных полей
        if not self.tms:
            errors.append("tms (case_id) обязателен")
        
        if not self.file_path:
            errors.append("file_path обязателен")
        
        # Проверка формата GUID (базовая проверка)
        if self.tms and not self._is_valid_guid(self.tms):
            errors.append(f"tms (case_id) должен быть валидным GUID, получено: {self.tms}")
        
        if self.suite_id and not self._is_valid_guid(self.suite_id):
            errors.append(f"suite_id должен быть валидным GUID, получено: {self.suite_id}")
        
        if errors:
            raise ParserValidationError(
                f"Ошибки валидации метаданных в файле {self.file_path}:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
    
    @staticmethod
    def _is_valid_guid(value: str) -> bool:
        """Проверка, что строка похожа на GUID"""
        if not value:
            return False
        
        # Базовая проверка формата GUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        parts = value.split('-')
        if len(parts) != 5:
            return False
        
        if len(parts[0]) != 8 or len(parts[1]) != 4 or len(parts[2]) != 4 or \
           len(parts[3]) != 4 or len(parts[4]) != 12:
            return False
        
        # Проверка, что все символы hex
        try:
            int(value.replace('-', ''), 16)
            return True
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'tms': self.tms,
            'file_path': self.file_path,
            'suite_id': self.suite_id,
            'suite_name': self.suite_name,
            'title': self.title,
            'description': self.description,
            'function_name': self.function_name,
            'severity': self.severity,
            'priority': self.priority,
            'tags': self.tags,
            'links': self.links,
            'steps': self.steps,
            'iterations': self.iterations,
            'suite_status': self.suite_status,
            'suite_description': self.suite_description,
            'suite_author': self.suite_author,
            'suite_created_at': self.suite_created_at,
            'estimated_time': self.estimated_time,
            'custom_fields': self.custom_fields,
        }


class BaseParser(ABC):
    """Базовый класс для всех парсеров"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Проверяет, может ли парсер обработать данный файл.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            True если парсер может обработать файл
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[TestMetadata]:
        """
        Парсит файл и извлекает метаданные тестов.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            Список метаданных тестов
            
        Raises:
            ParserError: при ошибках парсинга
            ParserValidationError: при ошибках валидации
        """
        pass
    
    def _map_severity_to_priority(self, severity: str) -> str:
        """Маппинг severity -> priority"""
        severity_lower = severity.lower()
        
        if severity_lower in ['blocker', 'critical', 'normal', 'minor', 'trivial']:
            return severity_lower
        else:
            return 'normal'
    
    def _parse_savetest_comments(self, lines: List[str]) -> Dict[str, str]:
        """
        Читает savetest_* комментарии из начала файла.
        Останавливается на первой непустой и не-комментарийной строке.
        Возвращает dict с ключами: status, author, description, created_at и др.
        """
        result: Dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if stripped == '':
                continue
            if not stripped.startswith('#'):
                break
            m = re.match(r'^#\s*savetest_(\w+):\s*(.+)$', stripped)
            if m:
                key = m.group(1)
                val = m.group(2).strip().replace('\\n', '\n')
                result[key] = val
        return result
    
    def _extract_case_savetest_comments(self, comment_lines: List[str]) -> Dict[str, str]:
        """
        Извлекает savetest_case_* метаданные (одиночные) из списка комментариев.
        Повторяющиеся ключи (напр. custom_field) не включаются — используйте _extract_case_custom_fields.
        """
        result: Dict[str, str] = {}
        for line in comment_lines:
            stripped = line.strip()
            m = re.match(r'^#\s*savetest_case_(\w+):\s*(.+)$', stripped)
            if m:
                key = m.group(1)
                if key == 'custom_field':
                    continue  # обрабатывается отдельно
                val = m.group(2).strip().replace('\\n', '\n')
                result[key] = val
        return result

    def _extract_case_custom_fields(self, comment_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Извлекает кастомные поля из savetest_case_custom_field: Name|Value комментариев.
        Возвращает список {'name': ..., 'value': ...}.
        """
        fields = []
        for line in comment_lines:
            stripped = line.strip()
            m = re.match(r'^#\s*savetest_case_custom_field:\s*(.+)$', stripped)
            if m:
                raw = m.group(1).strip()
                pipe_idx = raw.find('|')
                if pipe_idx < 0:
                    fields.append({'name': raw, 'value': ''})
                else:
                    fields.append({
                        'name': raw[:pipe_idx],
                        'value': raw[pipe_idx + 1:].replace('\\n', '\n')
                    })
        return fields
    
    def _validate_and_process_metadata(self, metadata_list: List[TestMetadata], file_path: Path) -> List[TestMetadata]:
        """Валидация и обработка списка метаданных."""
        if not metadata_list:
            return []
        
        # Валидация каждого элемента
        errors = []
        for i, metadata in enumerate(metadata_list):
            try:
                metadata.validate()
            except ParserValidationError as e:
                errors.append(f"Тест #{i+1}: {str(e)}")
        
        if errors:
            raise ParserValidationError(
                f"Ошибки валидации в файле {file_path}:\n" + "\n".join(errors)
            )
        
        # Проверка дубликатов case_id внутри файла
        case_ids = [m.tms for m in metadata_list]
        duplicates = [cid for cid in case_ids if case_ids.count(cid) > 1]
        if duplicates:
            unique_duplicates = list(set(duplicates))
            raise ParserValidationError(
                f"Файл {file_path} содержит дублирующиеся case_id (tms): {', '.join(unique_duplicates)}"
            )
        
        return metadata_list

    @staticmethod
    def check_cross_file_duplicates(
        metadata_list: List["TestMetadata"],
    ) -> tuple:
        """
        Проверяет дубликаты case_id среди метаданных из нескольких файлов.

        Используется в /parse_repo endpoint для обнаружения дубликатов до
        возврата результатов в экстрактор.

        Returns:
            (deduplicated_list, error_messages)
        """
        seen: Dict[str, str] = {}
        clean = []
        errors = []

        for m in metadata_list:
            if m.tms in seen:
                errors.append(
                    f"Дубликат case_id '{m.tms}': "
                    f"{seen[m.tms]} и {m.file_path}"
                )
            else:
                seen[m.tms] = m.file_path
                clean.append(m)

        return clean, errors

