"""ЛР 2. Общие данные: каталог лицензий, сценарии использования, профили дефектов, перечень рисков NIST AI 600-1.

Каталог лицензий — УЧЕБНАЯ упрощённая модель условий Creative Commons (BY, SA, NC, ND) по описаниям
creativecommons.org (проверено 19.09.2026). Это не юридическая консультация: реальные решения принимает юрист.
Идентификаторы — SPDX License List (версия 3.29.0 от 16.09.2026).
"""
from __future__ import annotations

# идентификатор SPDX -> условия
LICENSES = {
    "CC0-1.0":         {"attribution": False, "nc": False, "nd": False, "sa": False},
    "CC-BY-4.0":       {"attribution": True,  "nc": False, "nd": False, "sa": False},
    "CC-BY-SA-4.0":    {"attribution": True,  "nc": False, "nd": False, "sa": True},
    "CC-BY-NC-4.0":    {"attribution": True,  "nc": True,  "nd": False, "sa": False},
    "CC-BY-ND-4.0":    {"attribution": True,  "nc": False, "nd": True,  "sa": False},
    "CC-BY-NC-SA-4.0": {"attribution": True,  "nc": True,  "nd": False, "sa": True},
    "MIT":             {"attribution": True,  "nc": False, "nd": False, "sa": False},
    "Apache-2.0":      {"attribution": True,  "nc": False, "nd": False, "sa": False},
}

# сценарии (контексты вариантов): условия использования набора
SCENARIOS = {
    "poster":  ("афиша фестиваля (печать, продажа билетов)",
                {"commercial_use": True,  "redistribution": True,  "derivatives": True,  "share_alike_acceptable": False}),
    "podcast": ("обложки подкастов (публикация в сети, монетизация)",
                {"commercial_use": True,  "redistribution": True,  "derivatives": True,  "share_alike_acceptable": True}),
    "campus":  ("открытый учебный набор университета (некоммерческий)",
                {"commercial_use": False, "redistribution": True,  "derivatives": True,  "share_alike_acceptable": True}),
    "proto":   ("внутренний прототип без публикации",
                {"commercial_use": False, "redistribution": False, "derivatives": True,  "share_alike_acceptable": True}),
    "catalog": ("электронный каталог фотоархива (публикация без изменений)",
                {"commercial_use": False, "redistribution": True,  "derivatives": False, "share_alike_acceptable": False}),
}
# профили дефектов (факторы вариантов)
PROFILES = {
    "nc":      "часть материалов под лицензиями с условием NonCommercial",
    "sa":      "часть материалов под лицензией с условием ShareAlike",
    "nd":      "часть материалов под лицензией с условием NoDerivatives",
    "license": "лицензия не указана или записана нестандартно",
    "privacy": "персональные данные и разные виды согласий",
    "origin":  "пробелы в происхождении: источник, дата, дубликаты",
}

# 12 групп рисков генеративного ИИ по NIST AI 600-1 (раздел 2, названия как в документе)
NIST_RISKS = [
    "CBRN Information or Capabilities", "Confabulation", "Dangerous, Violent, or Hateful Content", "Data Privacy",
    "Environmental Impacts", "Harmful Bias or Homogenization", "Human-AI Configuration", "Information Integrity",
    "Information Security", "Intellectual Property", "Obscene, Degrading, and/or Abusive Content",
    "Value Chain and Component Integration",
]

MANIFEST_FIELDS = ["id", "filename", "media_type", "source", "date_acquired", "license", "rights_holder",
                   "personal_data", "consent", "sha256"]


def variant_to_pair(n: int) -> tuple[str, str]:
    c, f = divmod(n - 1, len(PROFILES))
    return list(SCENARIOS)[c], list(PROFILES)[f]
