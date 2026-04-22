from __future__ import annotations

import ast
import math
from datetime import datetime
from zoneinfo import ZoneInfo

MOSCOW_TZ = ZoneInfo('Europe/Moscow')
_ALLOWED_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError('Unsupported expression')


def _safe_eval(expression: str) -> str | None:
    expr = expression.replace(',', '.').strip()
    if not expr or len(expr) > 120:
        return None
    try:
        parsed = ast.parse(expr, mode='eval')
        value = _eval_ast(parsed)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    rounded = round(value, 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f'{rounded:.6f}'.rstrip('0').rstrip('.')


def generate_assistant_reply(text: str) -> str:
    cleaned = ' '.join((text or '').strip().split())
    lowered = cleaned.lower()

    if not cleaned:
        return 'Пожалуйста, введите текст или вопрос.'

    math_result = _safe_eval(cleaned)
    if math_result is not None:
        return f'Результат вычисления: {math_result}.'

    now = datetime.now(MOSCOW_TZ)

    if any(token in lowered for token in ['который час', 'сколько времени', 'текущее время', 'время']) and len(lowered) < 80:
        return f'Сейчас {now.strftime("%H:%M")} по московскому времени.'

    if any(token in lowered for token in ['какое сегодня число', 'сегодняшняя дата', 'какой сегодня день', 'дата']) and len(lowered) < 100:
        return f'Сегодня {now.strftime("%d.%m.%Y")}.'

    if any(token in lowered for token in ['привет', 'здравствуй', 'добрый день', 'добрый вечер']):
        return 'Привет! Я готов озвучить текст или ответить на простой запрос в демонстрационном режиме.'

    if 'кто ты' in lowered or 'что ты умеешь' in lowered:
        return (
            'Я демонстрационный голосовой помощник. Сейчас я умею озвучивать текст, '
            'выбирать движок синтеза и отвечать на простые офлайн-запросы: приветствие, дата, время и вычисления.'
        )

    if any(token in lowered for token in ['спасибо', 'благодарю']):
        return 'Пожалуйста!'

    return (
        'Я понял запрос. В текущей версии это офлайн-демо без большой языковой модели, '
        'поэтому я могу надежно озвучить текст и обработать только базовые команды.'
    )
