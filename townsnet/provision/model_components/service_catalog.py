"""Service catalog and naming utilities for provision model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

# Public mapping of services by thematic group.
def _normalize_service_key(value: str) -> str:
    """Prepare service names for case-insensitive lookup."""
    normalized = value.strip().lower().replace("ё", "е")
    normalized = re.sub(r"[\s_\-]+", " ", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

SERVICE_ID_TO_NAME: Dict[int, str] = {
    1: "Парк",
    21: "Детский сад",
    22: "Школа",
    23: "Дом детского творчества",
    25: "Детские лагеря",
    26: "Среднее специальное учебное заведение",
    27: "Высшее учебное заведение",
    28: "Поликлиника",
    29: "Детская поликлиника",
    30: "Стоматологическая клиника",
    31: "Фельдшерско-акушерский пункт",
    32: "Женская консультация",
    33: "Реабилитационный центр",
    34: "Аптека",
    35: "Больница",
    36: "Роддом",
    37: "Детская больница",
    38: "Хоспис",
    39: "Станция скорой медицинской помощи",
    40: "Травматологические пункты",
    41: "Морг",
    42: "Диспансер",
    44: "Дом престарелых",
    45: "Центр занятости населения",
    46: "Детские дома-интернаты",
    47: "Многофункциональные центры (МФЦ)",
    48: "Библиотека",
    49: "Дворец культуры",
    50: "Музей",
    51: "Театр",
    56: "Кинотеатр",
    57: "Торговый центр",
    58: "Аквапарк",
    59: "Стадион",
    60: "Ледовая арена",
    61: "Кафе",
    62: "Ресторан",
    63: "Бар/Паб",
    64: "Столовая",
    65: "Булочная",
    67: "Бассейн",
    68: "Спортивный зал",
    69: "Каток",
    73: "Скалодром",
    78: "Полицейский участок",
    79: "Пожарная станция",
    81: "Железнодорожный вокзал",
    86: "Автовокзал",
    88: "Выход метро",
    89: "Супермаркет",
    90: "Продукты (магазин у дома)",
    91: "Рынок",
    92: "Хозяйственные товары",
    93: "Одежда и обувь",
    94: "Бытовая техника",
    95: "Книжный магазин",
    96: "Детские товары",
    97: "Спортивный магазин",
    98: "Почтовое отделение",
    99: "Пункт выдачи",
    100: "Отделение банка",
    101: "Банкомат",
    102: "Адвокат",
    103: "Нотариальная контора",
    104: "Парикмахер",
    105: "Салон красоты",
    106: "Общественная баня",
    107: "Ветеринарная клиника",
    108: "Зоомагазин",
    110: "Гостиница",
    111: "Хостел",
    112: "База отдыха",
    113: "Памятник",
    114: "Церковь",
    143: "Санаторий",
}

SERVICE_NAME_ALIASES: Dict[str, str] = {
    _normalize_service_key(name): name for name in SERVICE_ID_TO_NAME.values()
}

SERVICE_GROUPS: Dict[str, List[str]] = {
    "Образование": [
        SERVICE_ID_TO_NAME[21],  # Детский сад
        SERVICE_ID_TO_NAME[22],  # Школа
        SERVICE_ID_TO_NAME[23],  # Дом детского творчества
        SERVICE_ID_TO_NAME[25],  # Детские лагеря
        SERVICE_ID_TO_NAME[26],  # Среднее специальное учебное заведение
        SERVICE_ID_TO_NAME[27],  # Высшее учебное заведение
    ],
    "Здравоохранение": [
        SERVICE_ID_TO_NAME[28],  # Поликлиника
        SERVICE_ID_TO_NAME[29],  # Детская поликлиника
        SERVICE_ID_TO_NAME[30],  # Стоматологическая клиника
        SERVICE_ID_TO_NAME[31],  # Фельдшерско-акушерский пункт
        SERVICE_ID_TO_NAME[32],  # Женская консультация
        SERVICE_ID_TO_NAME[33],  # Реабилитационный центр
        SERVICE_ID_TO_NAME[34],  # Аптека
        SERVICE_ID_TO_NAME[35],  # Больница
        SERVICE_ID_TO_NAME[36],  # Роддом
        SERVICE_ID_TO_NAME[37],  # Детская больница
        SERVICE_ID_TO_NAME[38],  # Хоспис
        SERVICE_ID_TO_NAME[39],  # Станция скорой медицинской помощи
        SERVICE_ID_TO_NAME[40],  # Травматологические пункты
        SERVICE_ID_TO_NAME[41],  # Морг
        SERVICE_ID_TO_NAME[42],  # Диспансер
        SERVICE_ID_TO_NAME[143], # Санаторий
    ],
    "Спорт": [
        SERVICE_ID_TO_NAME[67],  # Бассейн
        SERVICE_ID_TO_NAME[68],  # Спортивный зал
        SERVICE_ID_TO_NAME[59],  # Стадион
        SERVICE_ID_TO_NAME[60],  # Ледовая арена
        SERVICE_ID_TO_NAME[69],  # Каток
        SERVICE_ID_TO_NAME[73],  # Скалодром
    ],
    "Социальная помощь": [
        SERVICE_ID_TO_NAME[44],  # Дом престарелых
        SERVICE_ID_TO_NAME[46],  # Детские дома-интернаты
        SERVICE_ID_TO_NAME[45],  # Центр занятости населения
    ],
    "Услуги": [
        SERVICE_ID_TO_NAME[98],  # Почтовое отделение
        SERVICE_ID_TO_NAME[99],  # Пункт выдачи
        SERVICE_ID_TO_NAME[100], # Отделение банка
        SERVICE_ID_TO_NAME[101], # Банкомат
        SERVICE_ID_TO_NAME[47],  # Многофункциональные центры (МФЦ)
        SERVICE_ID_TO_NAME[102], # Адвокат
        SERVICE_ID_TO_NAME[103], # Нотариальная контора
        SERVICE_ID_TO_NAME[104], # Парикмахер
        SERVICE_ID_TO_NAME[105], # Салон красоты
        SERVICE_ID_TO_NAME[106], # Общественная баня
        SERVICE_ID_TO_NAME[107], # Ветеринарная клиника
        SERVICE_ID_TO_NAME[108], # Зоомагазин
    ],
    "Культура и отдых": [
        SERVICE_ID_TO_NAME[48],  # Библиотека
        SERVICE_ID_TO_NAME[49],  # Дворец культуры
        SERVICE_ID_TO_NAME[50],  # Музей
        SERVICE_ID_TO_NAME[51],  # Театр
        SERVICE_ID_TO_NAME[56],  # Кинотеатр
        SERVICE_ID_TO_NAME[57],  # Торговый центр
        SERVICE_ID_TO_NAME[58],  # Аквапарк
        SERVICE_ID_TO_NAME[59],  # Стадион (как объект массовых мероприятий)
        SERVICE_ID_TO_NAME[60],  # Ледовая арена
        SERVICE_ID_TO_NAME[1],   # Парк
    ],
    "Безопасность": [
        SERVICE_ID_TO_NAME[78],  # Полицейский участок
        SERVICE_ID_TO_NAME[79],  # Пожарная станция
    ],
    "Туризм": [
        SERVICE_ID_TO_NAME[110], # Гостиница
        SERVICE_ID_TO_NAME[111], # Хостел
        SERVICE_ID_TO_NAME[112], # База отдыха
        SERVICE_ID_TO_NAME[61],  # Кафе
        SERVICE_ID_TO_NAME[62],  # Ресторан
        SERVICE_ID_TO_NAME[63],  # Бар/Паб
        SERVICE_ID_TO_NAME[64],  # Столовая
        SERVICE_ID_TO_NAME[65],  # Булочная
    ],
}

_SERVICE_ID_PATTERN = re.compile(r"\d+")


def _canonical_service_name(candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    normalized = _normalize_service_key(str(candidate))
    return SERVICE_NAME_ALIASES.get(normalized)


def _extract_service_id(token: str) -> Optional[int]:
    matches = _SERVICE_ID_PATTERN.findall(token)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _service_name_from_id(service_id: int) -> Optional[str]:
    try:
        return SERVICE_ID_TO_NAME[int(service_id)]
    except (KeyError, ValueError):
        return None


def _service_name_from_path(path: Path) -> Optional[str]:
    service_id = _extract_service_id(path.stem)
    if service_id is None:
        return None
    return _service_name_from_id(service_id)



__all__ = [
    "SERVICE_GROUPS",
    "SERVICE_ID_TO_NAME",
    "SERVICE_NAME_ALIASES",
    "_normalize_service_key",
    "_canonical_service_name",
    "_extract_service_id",
    "_service_name_from_id",
    "_service_name_from_path",
]
