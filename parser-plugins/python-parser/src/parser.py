"""
Python Parser Plugin - FastAPI сервис для парсинга Python тестов
"""

import ast
import json
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Python Parser Plugin", version="1.0.0")


class ParseRequest(BaseModel):
    """Запрос на парсинг файла"""
    file_path: str
    repo_path: Optional[str] = None


class CanParseRequest(BaseModel):
    """Запрос на проверку возможности парсинга"""
    file_path: str


class ParseResponse(BaseModel):
    """Ответ с результатами парсинга"""
    success: bool
    metadata: List[Dict[str, Any]]
    error: Optional[str] = None


class CanParseResponse(BaseModel):
    """Ответ на проверку возможности парсинга"""
    can_parse: bool


class HealthResponse(BaseModel):
    """Статус плагина"""
    status: str
    plugin_name: str
    version: str


class PythonParser(BaseParser):
    """Парсер для Python файлов"""
    
    def can_parse(self, file_path: Path) -> bool:
        """Проверяет, является ли файл Python тестом"""
        name = file_path.name
        suffix = file_path.suffix
        
        if not suffix == '.py':
            return False
        
        can_parse = (name.startswith('test_') or 
                     name.endswith('_test.py') or 
                     name.startswith('Test'))
        
        return can_parse
    
    def parse_file(self, file_path: Path) -> List[TestMetadata]:
        """Парсит Python файл и извлекает Allure метаданные."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            raise ParserError(f"Не удалось прочитать файл {file_path}: {e}")
        
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.error(f"Синтаксическая ошибка в файле {file_path}: {e}")
            raise ParserError(f"Синтаксическая ошибка в файле {file_path}: {e}")
        except Exception as e:
            logger.error(f"Ошибка парсинга AST файла {file_path}: {e}")
            raise ParserError(f"Не удалось распарсить файл {file_path}: {e}")
        
        metadata_list = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                suite_id = self._get_suite_id_from_class(node)
                suite_name = self._get_suite_name_from_class(node)
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        is_test = item.name.startswith('test_') or self._has_test_decorator(item)
                        if is_test:
                            metadata = self._extract_from_function(item, str(file_path), suite_id, suite_name)
                            if metadata:
                                metadata_list.append(metadata)
            
            elif isinstance(node, ast.FunctionDef):
                is_test = node.name.startswith('test_') or self._has_test_decorator(node)
                if is_test:
                    metadata = self._extract_from_function(node, str(file_path), None, None)
                    if metadata:
                        metadata_list.append(metadata)
        
        return self._validate_and_process_metadata(metadata_list, file_path)
    
    def _get_suite_id_from_class(self, class_node: ast.ClassDef) -> Optional[str]:
        """Извлекает suite_id из @allure.suite() декоратора класса"""
        for decorator in class_node.decorator_list:
            decorator_info = self._parse_decorator(decorator)
            if decorator_info:
                decorator_name, decorator_args = decorator_info
                if decorator_name == 'allure.suite':
                    return decorator_args.get('value')
        return None
    
    def _get_suite_name_from_class(self, class_node: ast.ClassDef) -> Optional[str]:
        """Извлекает suite_name из @allure.story() декоратора класса"""
        for decorator in class_node.decorator_list:
            decorator_info = self._parse_decorator(decorator)
            if decorator_info:
                decorator_name, decorator_args = decorator_info
                if decorator_name == 'allure.story':
                    return decorator_args.get('value')
        return None
    
    def _has_test_decorator(self, node: ast.FunctionDef) -> bool:
        """Проверяет, есть ли у функции декораторы тестовых фреймворков"""
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            if decorator_name in ['pytest.mark', 'unittest', 'testng']:
                return True
        return False
    
    def _extract_from_function(
        self, 
        func_node: ast.FunctionDef, 
        file_path: str,
        suite_id: Optional[str],
        suite_name: Optional[str]
    ) -> Optional[TestMetadata]:
        """Извлекает метаданные из декораторов функции"""
        tms_value = None
        title_value = None
        description_value = None
        severity_value = None
        priority_value = None
        tags_list = []
        links_list = []
        iterations_list = []
        
        for decorator in func_node.decorator_list:
            parametrize_info = self._extract_parametrize(decorator)
            if parametrize_info:
                iterations_list.extend(parametrize_info)
                continue
            
            decorator_info = self._parse_decorator(decorator)
            
            if not decorator_info:
                continue
            
            decorator_name, decorator_args = decorator_info
            
            if decorator_name == 'allure.tms':
                tms_value = decorator_args.get('value')
            elif decorator_name == 'allure.title':
                title_value = decorator_args.get('value')
            elif decorator_name == 'allure.description':
                description_value = decorator_args.get('value')
            elif decorator_name == 'allure.severity':
                severity = decorator_args.get('value')
                if severity:
                    severity_value = severity
                    priority_value = self._map_severity_to_priority(severity)
            elif decorator_name == 'allure.tag':
                values = decorator_args.get('values', [])
                if values:
                    tags_list.extend(values)
            elif decorator_name == 'allure.issue':
                issue = decorator_args.get('value')
                if issue:
                    links_list.append({'name': issue, 'value': issue})
            elif decorator_name == 'allure.link':
                values = decorator_args.get('values')
                if values and len(values) >= 2:
                    links_list.append({'name': values[1], 'value': values[0]})
                elif values and len(values) == 1:
                    links_list.append({'name': 'link', 'value': values[0]})
                else:
                    link = decorator_args.get('value')
                    if link:
                        links_list.append({'name': 'link', 'value': link})
        
        if not tms_value:
            return None
        
        steps = self._extract_steps_from_function(func_node)
        
        return TestMetadata(
            tms=tms_value,
            file_path=file_path,
            suite_id=suite_id,
            suite_name=suite_name,
            title=title_value or func_node.name,
            description=description_value,
            function_name=func_node.name,
            severity=severity_value,
            priority=priority_value,
            tags=tags_list,
            links=links_list,
            steps=steps,
            iterations=iterations_list
        )
    
    def _parse_decorator(self, decorator) -> Optional[tuple]:
        """Парсит декоратор и возвращает (имя, аргументы)"""
        try:
            decorator_name = self._get_decorator_name(decorator)
            
            if not decorator_name:
                return None
            
            if not decorator_name.startswith('allure.'):
                return None
            
            args = {}
            
            if isinstance(decorator, ast.Call):
                if decorator.args:
                    values = []
                    for arg in decorator.args:
                        value = self._extract_static_value(arg)
                        if value is not None:
                            values.append(value)
                    
                    if len(values) == 1:
                        args['value'] = values[0]
                    elif len(values) > 1:
                        args['values'] = values
                
                for keyword in decorator.keywords:
                    value = self._extract_static_value(keyword.value)
                    if value is not None:
                        args[keyword.arg] = value
            
            return (decorator_name, args)
        
        except Exception:
            return None
    
    def _get_decorator_name(self, decorator) -> Optional[str]:
        """Получает полное имя декоратора"""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                parts = []
                node = decorator
                while isinstance(node, ast.Attribute):
                    parts.append(node.attr)
                    node = node.value
                if isinstance(node, ast.Name):
                    parts.append(node.id)
                return '.'.join(reversed(parts))
            elif isinstance(decorator, ast.Call):
                return self._get_decorator_name(decorator.func)
        except Exception:
            pass
        return None
    
    def _extract_static_value(self, node) -> Optional[str]:
        """Извлекает статическое значение из AST узла"""
        try:
            if isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Str):
                return node.s
            elif isinstance(node, ast.Num):
                return str(node.n)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Attribute) or isinstance(node.value, ast.Name):
                    return node.attr.lower()
        except Exception:
            pass
        return None
    
    def _extract_steps_from_function(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Извлекает шаги из with allure.step(...) в теле функции"""
        steps = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        call = item.context_expr
                        
                        func_name = None
                        if isinstance(call.func, ast.Attribute):
                            if (isinstance(call.func.value, ast.Name) and 
                                call.func.value.id == 'allure' and 
                                call.func.attr == 'step'):
                                func_name = 'allure.step'
                        elif isinstance(call.func, ast.Name):
                            if call.func.id == 'step':
                                func_name = 'step'
                        
                        if func_name:
                            step_name = None
                            if call.args and len(call.args) > 0:
                                step_name = self._extract_string_value(call.args[0])
                            
                            if step_name:
                                steps.append({
                                    'action': step_name,
                                    'expected_result': ''
                                })
        
        return steps
    
    def _extract_string_value(self, node) -> Optional[str]:
        """Извлекает строковое значение из AST узла"""
        try:
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    return node.value
            elif isinstance(node, ast.Str):
                return node.s
            elif isinstance(node, ast.JoinedStr):
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                    elif isinstance(value, ast.Str):
                        parts.append(value.s)
                    elif isinstance(value, ast.FormattedValue):
                        # Извлекаем выражение из f-строки, сохраняя шаблон переменной
                        expr_str = self._extract_formatted_expression(value.value)
                        parts.append(f"{{{expr_str}}}")
                return ''.join(parts)
        except Exception:
            pass
        return None
    
    def _extract_formatted_expression(self, node) -> str:
        """
        Извлекает строковое представление выражения из FormattedValue.
        Сохраняет структуру выражения в виде шаблона для переменных.
        
        Examples:
            - ast.Name('x') -> 'x'
            - ast.Attribute(value=Name('obj'), attr='attr') -> 'obj.attr'
            - ast.Call(func=Attribute(value=Name('obj'), attr='method')) -> 'obj.method()'
            - ast.BinOp(left=Name('a'), op=Add(), right=Name('b')) -> 'a + b'
        """
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                # Обработка атрибутов: obj.attr или obj.attr.subattr
                value_str = self._extract_formatted_expression(node.value)
                return f"{value_str}.{node.attr}"
            elif isinstance(node, ast.Call):
                # Обработка вызовов: obj.method() или func()
                func_str = self._extract_formatted_expression(node.func)
                args_strs = []
                for arg in node.args[:3]:  # Ограничиваем количество аргументов для читаемости
                    arg_str = self._extract_formatted_expression(arg)
                    args_strs.append(arg_str)
                if len(node.args) > 3:
                    args_strs.append("...")
                args_str = ", ".join(args_strs) if args_strs else ""
                return f"{func_str}({args_str})"
            elif isinstance(node, ast.BinOp):
                # Обработка бинарных операций: a + b, x - y и т.д.
                left_str = self._extract_formatted_expression(node.left)
                right_str = self._extract_formatted_expression(node.right)
                op_str = self._extract_operator(node.op)
                return f"{left_str} {op_str} {right_str}"
            elif isinstance(node, ast.UnaryOp):
                # Обработка унарных операций: -x, not y
                operand_str = self._extract_formatted_expression(node.operand)
                op_str = self._extract_unary_operator(node.op)
                return f"{op_str}{operand_str}"
            elif isinstance(node, ast.Subscript):
                # Обработка индексации: items[0], data['key'], arr[start:end]
                value_str = self._extract_formatted_expression(node.value)
                slice_str = self._extract_slice_expression(node.slice)
                return f"{value_str}[{slice_str}]"
            elif isinstance(node, ast.Constant):
                # Константные значения в выражениях
                if isinstance(node.value, str):
                    return f"'{node.value}'"
                return str(node.value)
            elif isinstance(node, ast.Str):
                # Для старых версий Python
                return f"'{node.s}'"
            elif isinstance(node, ast.Num):
                # Для старых версий Python
                return str(node.n)
            elif isinstance(node, ast.List):
                # Списки: [a, b, c]
                elts_strs = [self._extract_formatted_expression(elt) for elt in node.elts[:5]]
                if len(node.elts) > 5:
                    elts_strs.append("...")
                return f"[{', '.join(elts_strs)}]"
            elif isinstance(node, ast.Tuple):
                # Кортежи: (a, b, c)
                elts_strs = [self._extract_formatted_expression(elt) for elt in node.elts[:5]]
                if len(node.elts) > 5:
                    elts_strs.append("...")
                return f"({', '.join(elts_strs)})"
            elif isinstance(node, ast.Dict):
                # Словари: {'key': value}
                items_strs = []
                for i, (key, value) in enumerate(zip(node.keys[:3], node.values[:3])):
                    key_str = self._extract_formatted_expression(key) if key else "None"
                    value_str = self._extract_formatted_expression(value)
                    items_strs.append(f"{key_str}: {value_str}")
                if len(node.keys) > 3:
                    items_strs.append("...")
                return f"{{{', '.join(items_strs)}}}"
            else:
                # Для неизвестных типов узлов возвращаем обобщенное представление
                return type(node).__name__.lower().replace('ast.', '')
        except Exception:
            return "..."
    
    def _extract_operator(self, op_node) -> str:
        """Извлекает строковое представление бинарного оператора"""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
            ast.MatMult: "@",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
            ast.And: "and",
            ast.Or: "or",
        }
        return op_map.get(type(op_node), "...")
    
    def _extract_unary_operator(self, op_node) -> str:
        """Извлекает строковое представление унарного оператора"""
        op_map = {
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "not ",
            ast.Invert: "~",
        }
        return op_map.get(type(op_node), "")
    
    def _extract_slice_expression(self, slice_node) -> str:
        """Извлекает строковое представление среза для Subscript"""
        try:
            if isinstance(slice_node, ast.Slice):
                # Обработка срезов: [start:end:step]
                parts = []
                if slice_node.lower:
                    parts.append(self._extract_formatted_expression(slice_node.lower))
                if slice_node.upper or slice_node.step:
                    # Если есть upper или step, нужно добавить двоеточие
                    parts.append("")
                    if slice_node.upper:
                        parts.append(self._extract_formatted_expression(slice_node.upper))
                    if slice_node.step:
                        parts.append("")
                        parts.append(self._extract_formatted_expression(slice_node.step))
                # Объединяем части через двоеточие, убирая пустые строки в начале
                result = ":".join(parts).lstrip(":")
                return result if result else ":"
            elif isinstance(slice_node, ast.Tuple):
                # Обработка множественных индексов: arr[x, y]
                elts_strs = [self._extract_formatted_expression(elt) for elt in slice_node.elts]
                return ", ".join(elts_strs)
            else:
                # Простой индекс
                return self._extract_formatted_expression(slice_node)
        except Exception:
            return "..."
    
    def _extract_parametrize(self, decorator) -> List[Dict[str, Any]]:
        """Извлекает параметры из @pytest.mark.parametrize"""
        iterations = []
        
        try:
            if not isinstance(decorator, ast.Call):
                return iterations
            
            decorator_name = self._get_decorator_name(decorator)
            if not decorator_name or 'parametrize' not in decorator_name:
                return iterations
            
            if len(decorator.args) < 2:
                return iterations
            
            param_names_node = decorator.args[0]
            param_names_str = self._extract_string_value(param_names_node)
            if not param_names_str:
                return iterations
            
            param_names = [name.strip() for name in param_names_str.split(',')]
            
            values_node = decorator.args[1]
            if isinstance(values_node, (ast.List, ast.Tuple)):
                for idx, value_node in enumerate(values_node.elts):
                    iteration = {
                        'iteration_id': str(idx + 1),
                        'params': []
                    }
                    
                    if isinstance(value_node, (ast.List, ast.Tuple)):
                        for i, param_value_node in enumerate(value_node.elts):
                            if i < len(param_names):
                                param_value = self._extract_iteration_value(param_value_node)
                                iteration['params'].append({
                                    'name': param_names[i],
                                    'value': param_value
                                })
                    else:
                        param_value = self._extract_iteration_value(value_node)
                        if param_names:
                            iteration['params'].append({
                                'name': param_names[0],
                                'value': param_value
                            })
                    
                    iterations.append(iteration)
        
        except Exception:
            pass
        
        return iterations
    
    def _extract_iteration_value(self, node) -> str:
        """Извлекает значение параметра итерации"""
        try:
            if isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Str):
                return node.s
            elif isinstance(node, ast.Num):
                return str(node.n)
            elif isinstance(node, (ast.List, ast.Tuple)):
                values = []
                for item in node.elts:
                    val = self._extract_iteration_value(item)
                    if val:
                        values.append(val)
                return f"[{', '.join(values)}]"
        except Exception:
            pass
        return "..."


# Создаем единственный экземпляр парсера
parser = PythonParser()


@app.post("/parse", response_model=ParseResponse)
async def parse_file(request: ParseRequest):
    """Парсит файл и возвращает метаданные"""
    try:
        file_path = Path(request.file_path)
        
        if not file_path.exists():
            error_msg = f"Файл не найден: {file_path}"
            logger.error(error_msg)
            return ParseResponse(
                success=False,
                metadata=[],
                error=error_msg
            )
        
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


@app.post("/can_parse", response_model=CanParseResponse)
async def can_parse_file(request: CanParseRequest):
    """Проверяет, может ли парсер обработать файл"""
    try:
        file_path = Path(request.file_path)
        can_parse = parser.can_parse(file_path)
        return CanParseResponse(can_parse=can_parse)
    except Exception as e:
        logger.error(f"Ошибка проверки: {e}", exc_info=True)
        return CanParseResponse(can_parse=False)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка работоспособности плагина"""
    return HealthResponse(
        status="ok",
        plugin_name="python-parser",
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
        "plugin": "python-parser",
        "version": "1.0.0",
        "endpoints": ["/parse", "/can_parse", "/health", "/config"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

