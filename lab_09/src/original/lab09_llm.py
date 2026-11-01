# ЛР 9, часть 2: языковая модель как звено конвейера. Запросы, проверка схемы и покрытие смысловых элементов брифа.
# Только стандартная библиотека Python 3: работает в Engee, на ноутбуке и в Colab. Языковая модель — любой чат (в демонстрации — веб-версия Perplexity):
# запрос копируется в новый чат, ответ сохраняется в файл ответа целиком (текстом), файл проверяется score_file().
import json, re, sys

STYLES = ["flat vector illustration", "photorealistic", "3D render", "watercolor", "minimal geometric", "macro photography", "pencil sketch", "isometric 3D render"]
COMPOSITIONS = ["centered", "rule of thirds", "symmetrical", "top-down", "close-up", "wide negative space"]
PALETTE = ["blue", "amber", "orange", "teal", "green", "red", "pink", "purple", "yellow", "gray", "black", "white"]
AVOID = ["people", "text", "logo", "watermark", "blur"]
KEYS = ["subject", "setting", "style", "lighting", "palette", "composition", "avoid"]
# Какие запреты (avoid) содержит описание заказчика: ключ — (контекст, номер описания с 0); остальные описания запретов не содержат.
AVOID_EXPECTED = {(1, 0): ["people", "text"], (2, 2): ["people"], (3, 0): ["logo"], (4, 4): ["text"], (5, 0): ["logo"]}

# Пять контекстов по пять описаний заказчика. concepts — группы подстрок (латиница, нижний регистр): элемент брифа считается переданным,
# если в тексте JSON-объекта найдена хотя бы одна подстрока группы.
BRIEFS = {
    1: [
        ("Нужен плакат для фестиваля робототехники: светящаяся механическая рука в синих и янтарных тонах, без людей и текста.",
         [["arm", "hand"], ["glow", "lumin", "light"], ["blue"], ["amber", "orange"]]),
        ("Афиша соревнований роботов-манипуляторов: строгий индустриальный стиль, холодный свет, но только не синий; много пустого места для заголовка.",
         [["manipulator", "robot", "arm"], ["industrial", "technical"], ["cold", "cool", "white", "gray"], ["negative space", "empty"], ["!blue"]]),
        ("Обложка буклета школьного кружка робототехники: дружелюбный маленький робот на столе, тёплый свет лампы.",
         [["robot"], ["friendly", "cute", "small"], ["desk", "table"], ["warm", "lamp"]]),
        ("Баннер для сайта фестиваля: рука-манипулятор берёт светящийся шар, вид крупным планом; палитра — ровно два цвета: чёрный и электрический синий.",
         [["arm", "manipulator", "robot"], ["sphere", "ball", "orb"], ["glow", "lumin"], ["close-up", "close up"], ["black"], ["blue"], ["=2"]]),
        ("Иллюстрация к статье об автоматизации склада: ряды стеллажей и робот-погрузчик, изометрия, серо-оранжевая гамма.",
         [["warehouse", "shelv", "rack"], ["robot", "forklift", "loader"], ["isometric"], ["gray", "grey"], ["orange"]]),
    ],
    2: [
        ("Обложка подкаста о городе: слоистый горизонт из геометрических фигур, бирюзовый и оранжевый цвета, минимализм, квадратный формат.",
         [["skyline", "horizon", "city"], ["geometric"], ["teal", "turquoise"], ["orange"], ["minimal", "flat"]]),
        ("Обложка выпуска про ночной город: тёмное небо, огни окон, вид с крыши; оранжевого и янтарного в палитре быть не должно.",
         [["night", "dark"], ["window", "lights"], ["rooftop", "roof"], ["city"], ["!orange", "!amber"]]),
        ("Обложка выпуска о велодорожках и общественном транспорте: яркая плоская иллюстрация, без людей.",
         [["bicycle", "bike", "tram", "bus", "transport"], ["flat"], ["bright", "vivid", "colorful"]]),
        ("Обложка выпуска о дворах и зелёных крышах: вид сверху, тёплое утро; палитра — ровно два цвета: зелёный и жёлтый.",
         [["courtyard", "yard", "green roof", "rooftop"], ["top-down", "aerial", "above"], ["morning", "sunrise", "warm"], ["green"], ["yellow"], ["=2"]]),
        ("Обложка выпуска о старом районе: кирпичные фасады, закатный свет, акварель.",
         [["brick"], ["facade", "building"], ["sunset", "golden", "warm"], ["watercolor"]]),
    ],
    3: [
        ("Плакат научной выставки: созвездие соединённых узлов, тёмный фон, белые и голубые линии, без логотипов.",
         [["constellation", "node", "network"], ["connect", "line"], ["dark", "black"], ["white"], ["cyan", "light blue", "blue"]]),
        ("Афиша выставки о микромире: крупные кристаллы под микроскопом, холодная гамма, красного и оранжевого не использовать.",
         [["crystal"], ["microscop", "micro", "macro"], ["cold", "cool", "blue"], ["!red", "!orange"]]),
        ("Плакат о космосе: туманность и одинокий зонд, глубокий синий цвет, симметрия.",
         [["nebula"], ["probe", "spacecraft", "satellite"], ["blue"], ["symmetr"]]),
        ("Обложка каталога выставки: абстрактная спираль ДНК, светящиеся линии на тёмном фоне; цветов в палитре ровно три: голубой, фиолетовый, белый.",
         [["dna", "helix", "spiral"], ["glow", "lumin"], ["dark", "black"], ["blue"], ["purple"], ["white"], ["=3"]]),
        ("Постер об энергетике будущего: солнечные панели и ветряки на закате, широкая композиция.",
         [["solar", "panel"], ["wind", "turbine"], ["sunset"], ["wide", "panoram", "negative space"]]),
    ],
    4: [
        ("Визуал к лекциям по экологии: вид сверху на лес, слоистые зелёные формы, мягкий утренний свет.",
         [["forest", "canopy", "tree"], ["top-down", "above", "aerial"], ["green"], ["soft"], ["morning"]]),
        ("Слайд о загрязнении рек: чистая и мутная вода рядом, симметричная композиция, акварель, зелёного в палитре нет.",
         [["river", "water"], ["clean", "clear"], ["murky", "pollut", "dirty", "turbid"], ["symmetr"], ["watercolor"], ["!green"]]),
        ("Лекция о пчёлах и опылителях: пчела на цветке, макросъёмка, тёплый свет.",
         [["bee"], ["flower"], ["macro", "close-up"], ["warm"]]),
        ("Обложка курса о климате: тающий ледник, широкий план; палитра — ровно два цвета: синий и белый.",
         [["glacier", "ice"], ["melt"], ["wide", "panoram"], ["blue"], ["white"], ["=2"]]),
        ("Заставка о переработке отходов: аккуратные стопки цветных материалов, плоская иллюстрация, без текста.",
         [["recycl", "waste", "paper", "glass", "plastic", "material"], ["stack", "pile"], ["color"], ["flat"]]),
    ],
    5: [
        ("Обложка электронного релиза: шлифованный металл с радужными бликами, макросъёмка, сильный контраст, без логотипов.",
         [["metal", "brushed"], ["iridescent", "rainbow"], ["macro"], ["contrast"]]),
        ("Обложка синтвейв-альбома: неоновая сетка на горизонте, розовый и фиолетовый закат, жёлтого в палитре быть не должно.",
         [["neon"], ["grid"], ["horizon", "sunset"], ["pink"], ["purple", "violet"], ["!yellow"]]),
        ("Обложка ambient-трека: туман над тёмным озером, минимализм, одинокий огонёк.",
         [["fog", "mist"], ["lake", "water"], ["dark"], ["light", "lamp", "glow"], ["minimal"]]),
        ("Обложка техно-сингла: чёрно-белые геометрические полосы, высокий контраст, симметрия; цветов в палитре ровно два.",
         [["stripe", "line", "geometric"], ["black"], ["white"], ["contrast"], ["symmetr"], ["=2"]]),
        ("Обложка дрим-поп релиза: пастельные облака и стеклянные шары, мягкий свет.",
         [["cloud"], ["glass", "sphere", "ball", "orb"], ["pastel"], ["soft"]]),
    ],
}

TASK = ("Ты помощник дизайнера. Преобразуй пять описаний заказчика (на русском языке) в пять JSON-объектов для генератора изображений. "
        "Не используй поиск в интернете и не добавляй ссылки. Ответ: один JSON-массив из пяти объектов в одном блоке кода, в том же порядке, что описания. "
        "Поля каждого объекта — ровно эти семь, значения на английском языке: "
        "subject — предмет, 3–12 слов; setting — место или фон, 2–12 слов; "
        "style — одно из: " + ", ".join(STYLES) + "; lighting — освещение, 1–6 слов; palette — массив из 1–3 цветов ТОЛЬКО из списка: " + ", ".join(PALETTE) + "; "
        "composition — одно из: " + ", ".join(COMPOSITIONS) + "; avoid — массив, элементы ТОЛЬКО из списка: " + ", ".join(AVOID) + "; перечисли ровно то, что заказчик прямо запрещает («без …»); если запретов нет — пустой массив.")
EXAMPLES = (
    "Пример 1.\nОписание: Афиша джаз-фестиваля у моря: саксофон на закате, тёплые цвета, без текста.\n"
    '{"subject": "saxophone on a seaside pier", "setting": "sea coast at sunset", "style": "flat vector illustration", "lighting": "warm sunset light", '
    '"palette": ["orange", "purple"], "composition": "centered", "avoid": ["text"]}\n'
    "Пример 2.\nОписание: Открытка ко дню рождения: воздушные шары над полем, акварель, светлая палитра.\n"
    '{"subject": "colorful balloons floating above a field", "setting": "open green field, clear sky", "style": "watercolor", "lighting": "soft daylight", '
    '"palette": ["blue", "yellow", "pink"], "composition": "rule of thirds", "avoid": []}')
STEPS = ("Сначала для каждого описания кратко выпиши ключевые элементы (предмет, стиль, цвета, ограничения), затем дай итоговый JSON-массив в одном блоке кода.")


def build_prompt(approach, ctx):
    """approach: 'zero' | 'few' | 'step'. Возвращает текст запроса для нового чата."""
    descr = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(BRIEFS[ctx]))
    parts = [TASK]
    if approach == "few":
        parts.append("Два примера формата (для других описаний):\n" + EXAMPLES)
    if approach == "step":
        parts.append(STEPS)
    parts.append("Описания заказчика:\n" + descr)
    return "\n\n".join(parts)


def extract_array(text):
    """Последний блок кода с JSON-массивом; если блоков нет — последний массив верхнего уровня в тексте. None, если разобрать не удалось."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    cands = blocks[::-1] if blocks else []
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
    if m:
        cands.append(m.group(1))
    for c in cands:
        try:
            data = json.loads(c.strip())
        except ValueError:
            continue
        if isinstance(data, list):
            return data, bool(blocks)
    return None, bool(blocks)


def check_object(o):
    """Список нарушений схемы для одного объекта (пустой список — объект корректен)."""
    if not isinstance(o, dict):
        return ["не объект"]
    err = []
    if list(sorted(o.keys())) != sorted(KEYS):
        err.append("набор полей")
    nw = lambda s: len(str(s).split())
    if not isinstance(o.get("subject"), str) or not 3 <= nw(o.get("subject", "")) <= 12:
        err.append("subject 3–12 слов")
    if not isinstance(o.get("setting"), str) or not 2 <= nw(o.get("setting", "")) <= 12:
        err.append("setting 2–12 слов")
    if o.get("style") not in STYLES:
        err.append("style вне списка")
    if not isinstance(o.get("lighting"), str) or not 1 <= nw(o.get("lighting", "")) <= 6:
        err.append("lighting 1–6 слов")
    p = o.get("palette")
    if not (isinstance(p, list) and 1 <= len(p) <= 3 and all(x in PALETTE for x in p)):
        err.append("palette: 1–3 цвета из списка")
    if o.get("composition") not in COMPOSITIONS:
        err.append("composition вне списка")
    a = o.get("avoid")
    if not (isinstance(a, list) and all(x in AVOID for x in a)):
        err.append("avoid: элементы из списка")
    if re.search(r"[А-Яа-яЁё]", json.dumps(o, ensure_ascii=False)):
        err.append("кириллица")
    return err


def coverage(o, groups):
    """Доля выполненных групп. Обычная группа — в тексте объекта есть хотя бы одна подстрока; «!цвет» — этого цвета нет в palette;
    «=N» — в palette ровно N цветов."""
    text = json.dumps(o, ensure_ascii=False).lower()
    pal = o.get("palette") if isinstance(o, dict) and isinstance(o.get("palette"), list) else []
    ptxt = " ".join(str(x).lower() for x in pal)
    done = 0
    for g in groups:
        if g[0].startswith("!"):
            done += not any(s[1:] in ptxt for s in g)
        elif g[0].startswith("="):
            done += len(pal) == int(g[0][1:])
        else:
            done += any(s in text for s in g)
    return done / len(groups)


def score_text(text, ctx):
    """K1 — доля объектов из пяти, прошедших проверку схемы (0, если массив не разобран); K2 — среднее покрытие смысловых элементов брифа;
    K3 — доля объектов, у которых список avoid в точности совпал с запретами описания (порядок не важен)."""
    arr, fenced = extract_array(text)
    out = dict(parsed=arr is not None, fenced=fenced, n=0 if arr is None else len(arr), k1=0.0, k2=0.0, k3=0.0, errors=[])
    if arr is None or len(arr) != 5:
        out["errors"].append("массив не разобран" if arr is None else f"объектов {len(arr)} вместо 5")
        if arr is None:
            return out
    ok, cov, exact = 0, [], 0
    for i, o in enumerate(arr[:5]):
        e = check_object(o)
        ok += not e
        out["errors"].extend(f"объект {i + 1}: {x}" for x in e)
        cov.append(coverage(o, BRIEFS[ctx][i][1]))
        exact += isinstance(o, dict) and isinstance(o.get("avoid"), list) and sorted(o["avoid"]) == sorted(AVOID_EXPECTED.get((ctx, i), []))
    cov += [0.0] * (5 - len(cov))
    out["k1"], out["k2"], out["k3"] = ok / 5, sum(cov) / 5, exact / 5
    return out


def score_file(path, ctx):
    return score_text(open(path, encoding="utf-8").read(), ctx)


def stability(paths):
    """K4 — стабильность выбора между повторами одного приёма: доля совпадений полей style и composition по всем парам повторов и всем пяти объектам.
    Учитываются только ответы, из которых разобран массив из пяти объектов."""
    arrs = [a for a, _ in (extract_array(open(p, encoding="utf-8").read()) for p in paths) if a is not None and len(a) == 5]
    same = tot = 0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            for k in range(5):
                for field in ("style", "composition"):
                    same += arrs[i][k].get(field) == arrs[j][k].get(field)
                    tot += 1
    return same / tot if tot else float("nan")


if __name__ == "__main__":
    # python lab09_llm.py prompt zero 1   — напечатать запрос;  python lab09_llm.py score answer.txt 1 — оценить ответ
    if sys.argv[1] == "prompt":
        print(build_prompt(sys.argv[2], int(sys.argv[3])))
    else:
        print(json.dumps(score_file(sys.argv[2], int(sys.argv[3])), ensure_ascii=False, indent=1))
