from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from turkicdocgen.languages import canonical_language_mix

from .corpus_loader import load_corpus_records

if TYPE_CHECKING:
    import random

GENERATOR_SCHEMA_VERSION = "2.1"

DENSITY_STANDARD_THRESHOLD = 0.45
DENSITY_DENSE_THRESHOLD = 0.80
CHECKBOX_CHECKED_THRESHOLD = 0.35
SPECIAL_COVERAGE_TERMS = {
    "kk": (
        (
            "\u04d8\u043a\u0456\u043c\u0448\u0456\u043b\u0456\u043a",
            "\u04d9\u0434\u0456\u0441",
        ),
        (
            "\u0492\u044b\u043b\u044b\u043c\u0438",
            "\u0493\u0438\u043c\u0430\u0440\u0430\u0442",
        ),
        ("\u049a\u04b1\u0436\u0430\u0442", "\u049b\u04b1\u049b\u044b\u049b"),
        (
            "\u04a2-\u0438\u043d\u0434\u0435\u043a\u0441",
            "\u0436\u0430\u04a3\u0430\u043b\u044b\u049b",
        ),
        ("\u04e8\u0442\u0456\u043d\u0456\u0448", "\u04e9\u043d\u0456\u043c"),
        ("\u04b0\u043b\u0442\u0442\u044b\u049b", "\u04b1\u0439\u044b\u043c"),
        ("\u04ae\u043b\u0433\u0456", "\u04af\u043b\u0433\u0456"),
        ("\u04ba\u0438\u0436\u0440\u0430", "\u0433\u0430\u0443\u04bb\u0430\u0440"),
        ("\u0406\u0441", "\u0431\u0456\u043b\u0456\u043c"),
    ),
    "ky": (
        ("\u04a2-\u0431\u0435\u043b\u0433\u0438", "\u0436\u0430\u04a3\u044b"),
        (
            "\u04e8\u0442\u04af\u043d\u043c\u04e9",
            "\u04e9\u043d\u04af\u0433\u04af\u04af",
        ),
        ("\u04ae\u043b\u0433\u04af", "\u043a\u04af\u043d"),
    ),
}


@dataclass(frozen=True, slots=True)
class DocumentContext:
    record_index: int
    language: str
    organization: str
    department: str
    person_name: str
    recipient_name: str
    address: str
    phone: str
    email: str
    document_number: str
    date: str
    period: str
    subject: str


DENSITY_VARIANTS = ("standard", "dense", "extended")


def choose_density(rng: random.Random) -> str:
    pick = rng.random()
    if pick < DENSITY_STANDARD_THRESHOLD:
        return "standard"
    if pick < DENSITY_DENSE_THRESHOLD:
        return "dense"
    return "extended"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    key: str
    label: str
    value_type: str
    weight: float
    align: str = "left"


@dataclass(frozen=True, slots=True)
class TableSchema:
    schema_id: str
    title: str
    columns: tuple[ColumnSpec, ...]


@dataclass(frozen=True, slots=True)
class FormField:
    key: str
    label: str
    value_type: str


@dataclass(frozen=True, slots=True)
class FormSection:
    title: str
    fields: tuple[FormField, ...]


@dataclass(frozen=True, slots=True)
class FormSchema:
    schema_id: str
    title: str
    sections: tuple[FormSection, ...]


_DATA = {
    "kk": {
        "organizations": [
            "Астана қаласы білім басқармасы",
            "Аудандық қызмет көрсету орталығы",
            "Ұлттық зерттеу институты",
        ],
        "departments": [
            "Құжаттамалық қамтамасыз ету бөлімі",
            "Кадр қызметі",
            "Қаржы және есеп бөлімі",
        ],
        "names": [
            "Айдана Серікқызы",
            "Нұрлан Бақытұлы",
            "Әсел Қайратқызы",
            "Ержан Маратұлы",
            "Дана Ермекқызы",
            "Мақсат Айдосұлы",
            "Жанар Талғатқызы",
            "Арман Қанатұлы",
            "Салтанат Нұрқызы",
            "Бекзат Олжасұлы",
        ],
        "recipient": [
            "Бөлім басшысы Г. Әлиеваға",
            "Директор Б. Сәрсеновке",
            "Комиссия төрағасына",
        ],
        "addresses": [
            "Астана қ., Қабанбай батыр даңғылы, 12",
            "Алматы қ., Абай даңғылы, 48",
        ],
        "subjects": [
            "құжаттарды тіркеу туралы",
            "қызмет көрсету нәтижесі туралы",
            "өтінішті қарау туралы",
        ],
        "statuses": ["Қабылданды", "Орындалуда", "Аяқталды", "Қосымша мәлімет қажет"],
    },
    "ky": {
        "organizations": [
            "Бишкек шаарынын билим берүү башкармалыгы",
            "Райондук тейлөө борбору",
            "Улуттук изилдөө институту",
        ],
        "departments": [
            "Документтик камсыздоо бөлүмү",
            "Кадр кызматы",
            "Каржы жана эсеп бөлүмү",
        ],
        "names": [
            "Айжан Бакыт кызы",
            "Нурлан Азамат уулу",
            "Асель Кубат кызы",
            "Эрмек Талант уулу",
            "Бегайым Руслан кызы",
            "Азамат Кубаныч уулу",
            "Жылдыз Нурбек кызы",
            "Темирлан Эмил уулу",
            "Сезим Болот кызы",
            "Адилет Мирлан уулу",
        ],
        "recipient": [
            "Бөлүм башчысы Г. Алиевага",
            "Директор Б. Сарсеновго",
            "Комиссиянын төрагасына",
        ],
        "addresses": [
            "Бишкек ш., Чүй проспекти, 120",
            "Ош ш., Курманжан датка көчөсү, 18",
        ],
        "subjects": [
            "документтерди каттоо жөнүндө",
            "кызматтын жыйынтыгы жөнүндө",
            "арызды кароо жөнүндө",
        ],
        "statuses": [
            "Кабыл алынды",
            "Аткарылууда",
            "Аяктады",
            "Кошумча маалымат керек",
        ],
    },
}


def _base_language(language: str) -> str:
    return "ky" if canonical_language_mix(language) in {"ky", "ru_ky"} else "kk"


def bilingual(language: str, kk: str, ky: str, ru: str) -> str:
    lang = canonical_language_mix(language)
    local = ky if lang in {"ky", "ru_ky"} else kk
    return f"{local} / {ru}" if lang in {"ru_kk", "ru_ky"} else local


def build_document_context(
    language: str, index: int, rng: random.Random
) -> DocumentContext:
    lang = canonical_language_mix(language)
    data = _DATA[_base_language(lang)]
    year = 2022 + index % 5
    month = index % 12 + 1
    day = index * 7 % 27 + 1
    prefix = "Тіркеу №" if _base_language(lang) == "kk" else "Каттоо №"
    if lang in {"ru_kk", "ru_ky"}:
        prefix = f"{prefix} / Рег. №"
    if _base_language(lang) == "ky":
        phone = f"+996 {rng.choice([50, 55, 70, 77, 99])}{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
    else:
        phone = f"+7 7{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}"
    organization = rng.choice(data["organizations"])
    department = rng.choice(data["departments"])
    person_name = rng.choice(data["names"])
    recipient_name = rng.choice(data["recipient"])
    address = rng.choice(data["addresses"])
    base_subject = rng.choice(data["subjects"])
    base_language = _base_language(lang)
    coverage_terms = SPECIAL_COVERAGE_TERMS[base_language]
    upper_term, lower_term = coverage_terms[index % len(coverage_terms)]
    subject = f"{upper_term} {lower_term}: {base_subject}"
    return DocumentContext(
        record_index=index,
        language=lang,
        organization=organization,
        department=department,
        person_name=person_name,
        recipient_name=recipient_name,
        address=address,
        phone=phone,
        email=f"office{index % 97:02d}@example.org",
        document_number=f"{prefix} {year}-{1000 + index}",
        date=f"{day:02d}.{month:02d}.{year}",
        period=f"{month:02d}.{year}",
        subject=subject,
    )


def _columns(language: str, keys: tuple[str, ...]) -> tuple[ColumnSpec, ...]:
    labels = {
        "sequence": bilingual(language, "№", "№", "№"),
        "name": bilingual(language, "Аты-жөні", "Аты-жөнү", "ФИО"),
        "date": bilingual(language, "Күні", "Күнү", "Дата"),
        "doc": bilingual(language, "Құжат №", "Документ №", "Документ №"),
        "department": bilingual(language, "Бөлім", "Бөлүм", "Отдел"),
        "amount": bilingual(language, "Сома", "Сумма", "Сумма"),
        "status": bilingual(language, "Мәртебесі", "Статусу", "Статус"),
        "note": bilingual(language, "Ескертпе", "Эскертүү", "Примечание"),
        "item": bilingual(language, "Атауы", "Аталышы", "Наименование"),
        "score": bilingual(language, "Нәтиже", "Жыйынтык", "Результат"),
    }
    weights = {
        "sequence": 0.35,
        "name": 1.45,
        "date": 0.8,
        "doc": 1.0,
        "department": 1.25,
        "amount": 0.8,
        "status": 1.0,
        "note": 1.25,
        "item": 1.45,
        "score": 0.7,
    }
    aligns = {
        "sequence": "center",
        "date": "center",
        "amount": "right",
        "score": "right",
    }
    return tuple(
        ColumnSpec(key, labels[key], key, weights[key], aligns.get(key, "left"))
        for key in keys
    )


def table_schemas(language: str) -> tuple[TableSchema, ...]:
    return (
        TableSchema(
            "appeal_registry",
            bilingual(
                language,
                "Өтініштерді тіркеу журналы",
                "Арыздарды каттоо журналы",
                "Журнал регистрации обращений",
            ),
            _columns(language, ("sequence", "date", "doc", "name", "status", "note")),
        ),
        TableSchema(
            "employee_list",
            bilingual(
                language,
                "Қызметкерлер тізімі",
                "Кызматкерлердин тизмеси",
                "Список сотрудников",
            ),
            _columns(language, ("sequence", "name", "department", "status", "note")),
        ),
        TableSchema(
            "document_log",
            bilingual(
                language,
                "Құжаттар қозғалысы",
                "Документтердин кыймылы",
                "Журнал документов",
            ),
            _columns(
                language, ("sequence", "date", "doc", "department", "status", "note")
            ),
        ),
        TableSchema(
            "academic_results",
            bilingual(
                language, "Оқу нәтижелері", "Окуу жыйынтыктары", "Результаты обучения"
            ),
            _columns(language, ("sequence", "name", "department", "score", "status")),
        ),
        TableSchema(
            "inventory",
            bilingual(
                language, "Мүлік тізімдемесі", "Мүлк тизмеси", "Инвентарная ведомость"
            ),
            _columns(language, ("sequence", "item", "doc", "amount", "status", "note")),
        ),
        TableSchema(
            "expense_register",
            bilingual(
                language, "Шығыстар тізілімі", "Чыгымдардын реестри", "Реестр расходов"
            ),
            _columns(
                language, ("sequence", "date", "doc", "department", "amount", "note")
            ),
        ),
        TableSchema(
            "registry_extract",
            bilingual(
                language, "Реестрден үзінді", "Реестрден көчүрмө", "Выписка из реестра"
            ),
            _columns(language, ("sequence", "date", "doc", "name", "status")),
        ),
        TableSchema(
            "syllabus",
            bilingual(language, "Оқу жоспары", "Окуу планы", "Учебный план"),
            _columns(language, ("sequence", "item", "amount", "score")),
        ),
        TableSchema(
            "catalog_entry",
            bilingual(
                language, "Каталог жазбалары", "Каталог жазуулары", "Каталожные записи"
            ),
            _columns(language, ("sequence", "item", "doc", "note")),
        ),
        TableSchema(
            "invoice_like",
            bilingual(
                language,
                "Төлем шот-фактурасы",
                "Төлөм эсеп-фактурасы",
                "Счет-фактура на оплату",
            ),
            _columns(language, ("sequence", "item", "amount", "note")),
        ),
        TableSchema(
            "schedule_table",
            bilingual(
                language,
                "Сабақ кестесі",
                "Сабактардын расписаниеси",
                "Расписание занятий",
            ),
            _columns(language, ("sequence", "date", "item", "note")),
        ),
        TableSchema(
            "attendance_sheet",
            bilingual(
                language,
                "Қатысуды есепке алу журналы",
                "Катышууну эсепке алуу журналы",
                "Журнал учета посещаемости",
            ),
            _columns(language, ("sequence", "name", "date", "status", "note")),
        ),
    )


def form_schemas(language: str) -> tuple[FormSchema, ...]:
    def field(
        key: str, kk: str, ky: str, ru: str, value_type: str = "text"
    ) -> FormField:
        return FormField(key, bilingual(language, kk, ky, ru), value_type)

    common = (
        FormSection(
            bilingual(language, "Өтініш беруші", "Арыз берүүчү", "Заявитель"),
            (
                field("name", "Аты-жөні", "Аты-жөнү", "ФИО"),
                field("id", "ЖСН", "Жеке номер", "Идентификатор"),
                field("address", "Мекенжайы", "Дареги", "Адрес"),
                field("phone", "Телефон", "Телефон", "Телефон", "phone"),
                field(
                    "email",
                    "Электрондық пошта",
                    "Электрондук почта",
                    "Электронная почта",
                    "email",
                ),
            ),
        ),
        FormSection(
            bilingual(
                language,
                "Құжат мәліметтері",
                "Документтин маалыматы",
                "Сведения о документе",
            ),
            (
                field(
                    "doc_number", "Құжат нөмірі", "Документ номери", "Номер документа"
                ),
                field("date", "Күні", "Күнү", "Дата", "date"),
                field("department", "Бөлім", "Бөлүм", "Отдел"),
                field(
                    "request_type",
                    "Өтініш түрі",
                    "Арыздын түрү",
                    "Вид обращения",
                    "choice",
                ),
                field(
                    "position",
                    "Лауазымы",
                    "Кызматы",
                    "Должность",
                ),
                field(
                    "preferred_contact",
                    "Байланыс тәсілі",
                    "Байланыш ыкмасы",
                    "Способ связи",
                    "choice",
                ),
                field(
                    "request_summary",
                    "Өтініштің қысқаша мазмұны",
                    "Арыздын кыскача мазмуну",
                    "Краткое содержание обращения",
                    "multiline",
                ),
            ),
        ),
        FormSection(
            bilingual(
                language,
                "Қосымшалар және келісім",
                "Тиркемелер жана макулдук",
                "Приложения и согласие",
            ),
            (
                field(
                    "attachments",
                    "Қосымша құжаттар",
                    "Кошумча документтер",
                    "Приложенные документы",
                    "checkbox",
                ),
                field(
                    "consent",
                    "Деректерді өңдеуге келісім",
                    "Маалыматты иштетүүгө макулдук",
                    "Согласие на обработку данных",
                    "checkbox",
                ),
                field(
                    "attachment_count",
                    "Қосымша парақ саны",
                    "Тиркеме барактарынын саны",
                    "Количество листов приложения",
                    "count",
                ),
            ),
        ),
        FormSection(
            bilingual(
                language, "Қызметтік бөлік", "Кызматтык бөлүк", "Служебная часть"
            ),
            (
                field(
                    "registry",
                    "Тіркеу нөмірі",
                    "Каттоо номери",
                    "Регистрационный номер",
                ),
                field("office_status", "Мәртебесі", "Статусу", "Статус", "choice"),
                field("executor", "Орындаушы", "Аткаруучу", "Исполнитель"),
                field(
                    "review_note",
                    "Қызметтік ескерту",
                    "Кызматтык эскертүү",
                    "Служебная отметка",
                    "note",
                ),
            ),
        ),
    )
    return (
        FormSchema(
            "service_request",
            bilingual(
                language,
                "Қызмет алуға өтініш",
                "Кызмат алуу арызы",
                "Заявление на получение услуги",
            ),
            common,
        ),
        FormSchema(
            "document_registration",
            bilingual(
                language,
                "Құжатты тіркеу нысаны",
                "Документти каттоо формасы",
                "Форма регистрации документа",
            ),
            common,
        ),
        FormSchema(
            "personnel_application",
            bilingual(language, "Кадрлық өтініш", "Кадрдык арыз", "Кадровое заявление"),
            common,
        ),
        FormSchema(
            "application_form",
            bilingual(language, "Анкета-өтініш", "Анкета-арыз", "Анкета-заявление"),
            common,
        ),
        FormSchema(
            "exam_sheet",
            bilingual(
                language, "Емтихан парағы", "Экзамен барагы", "Экзаменационный лист"
            ),
            common,
        ),
        FormSchema(
            "worksheet",
            bilingual(language, "Жұмыс парағы", "Иш барагы", "Рабочий лист"),
            common,
        ),
        FormSchema(
            "receipt_like",
            bilingual(
                language,
                "Төлем квитанциясы",
                "Төлөм квитанциясы",
                "Квитанция об оплате",
            ),
            common,
        ),
    )


def _row_date(base_date: str, row: int) -> str:
    parsed = datetime.strptime(base_date, "%d.%m.%Y")
    return (parsed + timedelta(days=row * 3)).strftime("%d.%m.%Y")


_KEY_HANDLERS = {
    "name": lambda key, ctx, r, rng, data: data["names"][
        (r + ctx.record_index * 7) % len(data["names"])
    ],
    "date": lambda key, ctx, r, rng, data: _row_date(ctx.date, r),
    "doc_number": lambda key, ctx, r, rng, data: (
        f"{_row_date(ctx.date, r)[-4:]}/{1000 + ctx.record_index * 19 + r * 17}"
    ),
    "department": lambda key, ctx, r, rng, data: data["departments"][
        (r + ctx.record_index * 3) % len(data["departments"])
    ],
    "office_status": lambda key, ctx, r, rng, data: data["statuses"][
        (r + ctx.record_index) % len(data["statuses"])
    ],
    "id": lambda key, ctx, r, rng, data: (
        f"{800000000000 + ctx.record_index * 1009 + r * 173:012d}"
    ),
    "address": lambda key, ctx, r, rng, data: ctx.address,
    "registry": lambda key, ctx, r, rng, data: ctx.document_number,
    "executor": lambda key, ctx, r, rng, data: ctx.person_name,
    "request_type": lambda key, ctx, r, rng, data: ctx.subject,
    "position": lambda key, ctx, r, rng, data: ctx.department,
    "preferred_contact": lambda key, ctx, r, rng, data: rng.choice(
        (ctx.phone, ctx.email)
    ),
    "attachments": lambda key, ctx, r, rng, data: bilingual(
        ctx.language,
        "Жеке куәлік көшірмесі",
        "Паспорттун көчүрмөсү",
        "Копия удостоверения",
    ),
}


def _handle_item(
    key: str, ctx: DocumentContext, r: int, rng: random.Random, data: dict
) -> str:
    items = (
        [
            "Ноутбук",
            "Принтер",
            "Кеңсе орындығы",
            "Құжат шкафы",
            "Монитор",
            "Сканер",
            "Пернетақта",
            "Желілік құрылғы",
            "Қағаз жинағы",
            "Мұрағат жәшігі",
        ]
        if _base_language(ctx.language) == "kk"
        else [
            "Ноутбук",
            "Принтер",
            "Кеңсе отургучу",
            "Документ шкафы",
            "Монитор",
            "Сканер",
            "Баскычтоп",
            "Тармак түзмөгү",
            "Кагаз топтому",
            "Архив кутусу",
        ]
    )
    return items[(r + ctx.record_index * 3) % len(items)]


def _handle_note(
    key: str, ctx: DocumentContext, r: int, rng: random.Random, data: dict
) -> str:
    notes = (
        (
            ("Тексерілді", "Проверено"),
            ("Қабылданды", "Принято"),
            ("Келісілді", "Согласовано"),
            ("Тіркелді", "Зарегистрировано"),
            ("Өңделді", "Обработано"),
            ("Жіберілді", "Направлено"),
            ("Нақтыланды", "Уточнено"),
            ("Мұрағатталды", "Архивировано"),
        )
        if _base_language(ctx.language) == "kk"
        else (
            ("Текшерилди", "Проверено"),
            ("Кабыл алынды", "Принято"),
            ("Макулдашылды", "Согласовано"),
            ("Катталды", "Зарегистрировано"),
            ("Иштелди", "Обработано"),
            ("Жөнөтүлдү", "Направлено"),
            ("Такталды", "Уточнено"),
            ("Архивделди", "Архивировано"),
        )
    )
    local, russian = notes[(r + ctx.record_index * 5) % len(notes)]
    return (
        f"{local} / {russian}"
        if canonical_language_mix(ctx.language)
        in {
            "ru_kk",
            "ru_ky",
        }
        else local
    )


_FIELD_TYPE_HANDLERS = {
    "doc": lambda key, ctx, r, rng, data: f"{2024 + r % 3}/{1000 + r * 17}",
    "department": lambda key, ctx, r, rng, data: data["departments"][
        (r + ctx.record_index * 3) % len(data["departments"])
    ],
    "amount": lambda key, ctx, r, rng, data: (
        f"{(r + 2 + ctx.record_index % 11) * 12500:,.2f}".replace(",", " ")
    ),
    "score": lambda key, ctx, r, rng, data: str(
        60 + (r * 7 + ctx.record_index * 3) % 41
    ),
    "status": lambda key, ctx, r, rng, data: data["statuses"][
        (r + ctx.record_index) % len(data["statuses"])
    ],
    "sequence": lambda key, ctx, r, rng, data: str(r + 1),
    "item": _handle_item,
    "phone": lambda key, ctx, r, rng, data: ctx.phone,
    "email": lambda key, ctx, r, rng, data: ctx.email,
    "date": lambda key, ctx, r, rng, data: _row_date(ctx.date, r),
    "checkbox": lambda key, ctx, r, rng, data: (
        "[x]" if rng.random() > CHECKBOX_CHECKED_THRESHOLD else "[ ]"
    ),
    "count": lambda key, ctx, r, rng, data: str(1 + (r + ctx.record_index) % 12),
    "multiline": lambda key, ctx, r, rng, data: ctx.subject.capitalize(),
    "note": _handle_note,
}


def value_for(
    field_type: str, key: str, context: DocumentContext, row: int, rng: random.Random
) -> str:
    data = _DATA[_base_language(context.language)]
    if key in ("name", "date"):
        return _KEY_HANDLERS[key](key, context, row, rng, data)
    if key == "doc_number" or field_type == "doc":
        return _KEY_HANDLERS["doc_number"](key, context, row, rng, data)
    if key == "department" or field_type == "department":
        return _KEY_HANDLERS["department"](key, context, row, rng, data)
    if field_type in ("amount", "score"):
        return _FIELD_TYPE_HANDLERS[field_type](key, context, row, rng, data)
    if field_type == "status" or key == "office_status":
        return _FIELD_TYPE_HANDLERS["status"](key, context, row, rng, data)
    if field_type in (
        "sequence",
        "item",
        "phone",
        "email",
        "date",
        "checkbox",
        "count",
        "multiline",
    ):
        return _FIELD_TYPE_HANDLERS[field_type](key, context, row, rng, data)
    if key in _KEY_HANDLERS:
        return _KEY_HANDLERS[key](key, context, row, rng, data)
    if field_type in _FIELD_TYPE_HANDLERS:
        return _FIELD_TYPE_HANDLERS[field_type](key, context, row, rng, data)
    return context.subject


def _populate_large_pools() -> None:
    import random

    local_rng = random.Random(42)

    kk_m_first = [
        "Нұрлан",
        "Ержан",
        "Мақсат",
        "Арман",
        "Бекзат",
        "Әлібек",
        "Бауыржан",
        "Руслан",
        "Дәурен",
        "Азамат",
        "Самат",
        "Қанат",
        "Досжан",
        "Мұрат",
        "Талғат",
        "Олжас",
        "Айдын",
        "Дархан",
        "Болат",
        "Елдос",
        "Асылбек",
        "Жандос",
        "Нұрсұлтан",
        "Тимур",
    ]
    kk_m_last = [
        "Бақытұлы",
        "Маратұлы",
        "Айдосұлы",
        "Қанатұлы",
        "Олжасұлы",
        "Серікұлы",
        "Асқарұлы",
        "Мұратұлы",
        "Нұрланұлы",
        "Болатұлы",
        "Дәулетұлы",
        "Саматұлы",
        "Талғатұлы",
        "Русланұлы",
        "Ержанұлы",
        "Әбілұлы",
        "Қайратұлы",
        "Нұрлыбекұлы",
    ]
    kk_f_first = [
        "Айдана",
        "Әсел",
        "Дана",
        "Жанар",
        "Салтанат",
        "Динара",
        "Анар",
        "Гүлнар",
        "Әлия",
        "Мәдина",
        "Аружан",
        "Бота",
        "Индира",
        "Зарина",
        "Сания",
        "Жұлдыз",
        "Арайлым",
        "Шынар",
        "Назерке",
        "Балжан",
        "Айгерім",
        "Меруерт",
        "Жазира",
        "Ләззат",
        "Дина",
    ]
    kk_f_last = [
        "Серікқызы",
        "Қайратқызы",
        "Ермекқызы",
        "Талғатқызы",
        "Нұрқызы",
        "Әлібекқызы",
        "Маратқызы",
        "Болатқызы",
        "Асқарқызы",
        "Дәулетқызы",
        "Саматқызы",
        "Қанатқызы",
        "Русланқызы",
        "Ержанқызы",
        "Айдосқызы",
    ]

    kk_names = []
    for m in kk_m_first:
        for ml in kk_m_last:
            kk_names.append(f"{m} {ml}")
    for f in kk_f_first:
        for fl in kk_f_last:
            kk_names.append(f"{f} {fl}")
    local_rng.shuffle(kk_names)
    _DATA["kk"]["names"] = kk_names[:400]

    kk_cities = [
        "Астана",
        "Алматы",
        "Шымкент",
        "Қарағанды",
        "Ақтөбе",
        "Тараз",
        "Павлодар",
        "Өскемен",
        "Семей",
        "Орал",
        "Қостанай",
        "Атырау",
        "Ақтау",
        "Қызылорда",
        "Түркістан",
        "Көкшетау",
    ]
    kk_org_types = [
        "қаласының білім басқармасы",
        "мемлекеттік университеті",
        "ауруханасы",
        "даму орталығы",
        "ғылыми-зерттеу институты",
        "коммуналдық кәсіпорны",
        "департаменті",
        "басқармасы",
        "ұлттық компаниясы",
        "оқу орталығы",
    ]
    kk_orgs = []
    for c in kk_cities:
        for t in kk_org_types:
            kk_orgs.append(f"{c} {t}")
    local_rng.shuffle(kk_orgs)
    _DATA["kk"]["organizations"] = kk_orgs[:150]

    kk_dep_names = [
        "Құжаттамалық қамтамасыз ету бөлімі",
        "Кадр қызметі",
        "Қаржы және есеп бөлімі",
        "Ақпараттық технологиялар бөлімі",
        "Заң қызметі",
        "Жоспарлау және талдау бөлімі",
        "Әкімшілік-шаруашылық бөлімі",
        "Ішкі аудит қызметі",
        "Стратегиялық даму бөлімі",
        "Қоғаммен байланыс бөлімі",
    ]
    _DATA["kk"]["departments"] = kk_dep_names

    kk_streets = [
        "Қабанбай батыр даңғылы",
        "Абай даңғылы",
        "Төле би көшесі",
        "Сәтбаев көшесі",
        "Мәңгілік Ел даңғылы",
        "Достық даңғылы",
        "Желтоқсан көшесі",
        "Бөгенбай батыр даңғылы",
        "Кенесары көшесі",
        "Сейфуллин даңғылы",
        "Байтұрсынов көшесі",
        "Рысқұлов даңғылы",
        "Шәкәрім көшесі",
    ]
    kk_addresses = []
    for c in kk_cities[:6]:
        for s in kk_streets:
            for h in range(1, 40):
                kk_addresses.append(f"{c} қ., {s}, {h}-үй")
    local_rng.shuffle(kk_addresses)
    _DATA["kk"]["addresses"] = kk_addresses[:300]

    kk_subj_verbs = [
        "құжаттарды тіркеу",
        "қызметтік есепті тапсыру",
        "өтінішті қарау",
        "келісімшартты бекіту",
        "бюджетті жоспарлау",
        "іс-шараны ұйымдастыру",
        "есепті талдау",
        "жобаны іске асыру",
        "материалдарды дайындау",
        "шешімді қабылдау",
    ]
    kk_subj_topics = [
        "туралы",
        "жөнінде",
        "мәселесі бойынша",
        "аясында",
        "нәтижелері туралы",
    ]
    kk_subjects = []
    for v in kk_subj_verbs:
        for t in kk_subj_topics:
            kk_subjects.append(f"{v} {t}")
    for s in _DATA["kk"]["subjects"]:
        kk_subjects.append(s)
    local_rng.shuffle(kk_subjects)
    _DATA["kk"]["subjects"] = kk_subjects[:300]

    kk_recipients = []
    for n in _DATA["kk"]["names"][:50]:
        parts = n.split()
        if len(parts) == 2:
            first, last = parts
            kk_recipients.append(f"Бас директор {first} {last} мырзаға")
            kk_recipients.append(f"Бөлім басшысы {first} {last}-ға")
    _DATA["kk"]["recipient"] = kk_recipients[:100]

    ky_m_first = [
        "Нурлан",
        "Эрмек",
        "Азамат",
        "Темирлан",
        "Адилет",
        "Бакыт",
        "Руслан",
        "Улан",
        "Данияр",
        "Канат",
        "Самат",
        "Алмаз",
        "Талант",
        "Нурбек",
        "Кубат",
        "Болот",
        "Мирлан",
        "Эмил",
        "Аскар",
        "Бектур",
        "Султан",
        "Чингиз",
        "Марат",
        "Кайрат",
        "Дастан",
    ]
    ky_m_last = [
        "Азамат уулу",
        "Талант уулу",
        "Кубаныч уулу",
        "Эмил уулу",
        "Мирлан уулу",
        "Бакыт уулу",
        "Руслан уулу",
        "Канат уулу",
        "Нурбек уулу",
        "Болот уулу",
        "Алмаз уулу",
        "Улан уулу",
        "Кубат уулу",
        "Аскар уулу",
        "Султан уулу",
    ]
    ky_f_first = [
        "Айжан",
        "Асель",
        "Бегайым",
        "Жылдыз",
        "Сезим",
        "Динара",
        "Гулнара",
        "Айдай",
        "Бермет",
        "Сайкал",
        "Каныкей",
        "Чолпон",
        "Айсулуу",
        "Назира",
        "Наргиза",
        "Эльнура",
        "Малика",
        "Алина",
        "Мээрим",
        "Кундуз",
        "Айсулу",
        "Жамиля",
        "Айжамал",
        "Дина",
        "Аида",
    ]
    ky_f_last = [
        "Бакыт кызы",
        "Кубат кызы",
        "Руслан кызы",
        "Нурбек кызы",
        "Болот кызы",
        "Мирлан кызы",
        "Азамат кызы",
        "Талант кызы",
        "Канат кызы",
        "Алмаз кызы",
        "Улан кызы",
        "Эмил кызы",
        "Аскар кызы",
        "Султан кызы",
    ]

    ky_names = []
    for m in ky_m_first:
        for ml in ky_m_last:
            ky_names.append(f"{m} {ml}")
    for f in ky_f_first:
        for fl in ky_f_last:
            ky_names.append(f"{f} {fl}")
    local_rng.shuffle(ky_names)
    _DATA["ky"]["names"] = ky_names[:400]

    ky_cities = [
        "Бишкек",
        "Ош",
        "Жалал-Абад",
        "Каракол",
        "Нарын",
        "Талас",
        "Баткен",
        "Токмок",
        "Чолпон-Ата",
        "Кант",
    ]
    ky_org_types = [
        "шаарынын билим берүү башкармалыгы",
        "мамлекеттик университети",
        "бейтапканасы",
        "өнүктүрүү борбору",
        "илимий-изилдөө институту",
        "муниципалдык ишканасы",
        "департаменти",
        "башкармалыгы",
        "улуттук компаниясы",
        "окуу борбору",
    ]
    ky_orgs = []
    for c in ky_cities:
        for t in ky_org_types:
            ky_orgs.append(f"{c} {t}")
    local_rng.shuffle(ky_orgs)
    _DATA["ky"]["organizations"] = ky_orgs[:150]

    ky_dep_names = [
        "Документтик камсыздоо бөлүмү",
        "Кадр кызматы",
        "Каржы жана эсеп бөлүмү",
        "Маалыматтык технологиялар бөлүмү",
        "Юридикалык кызмат",
        "Пландоо жана талдоо бөлүмү",
        "Административдик-чарбалык бөлүм",
        "Ички аудит кызматы",
        "Стратегиялык өнүктүрүү бөлүмү",
        "Коомчулук менен байланыш бөлүмү",
    ]
    _DATA["ky"]["departments"] = ky_dep_names

    ky_streets = [
        "Чүй проспекти",
        "Курманжан датка көчөсү",
        "Манас проспекти",
        "Киев көчөсү",
        "Токтогул көчөсү",
        "Абдрахманов көчөсү",
        "Байтик Баатыр көчөсү",
        "Московская көчөсү",
        "Фрунзе көчөсү",
        "Ахунбаев көчөсү",
        "Ибраимов көчөсү",
        "Жибек Жолу проспекти",
    ]
    ky_addresses = []
    for c in ky_cities[:6]:
        for s in ky_streets:
            for h in range(1, 40):
                ky_addresses.append(f"{c} ш., {s}, {h}-үй")
    local_rng.shuffle(ky_addresses)
    _DATA["ky"]["addresses"] = ky_addresses[:300]

    ky_subj_verbs = [
        "документтерди каттоо",
        "кызматтык отчетту тапшыруу",
        "арызды кароо",
        "келишимди бекитүү",
        "бюджетти пландоо",
        "иш-чараны уюштуруу",
        "отчетту талдоо",
        "долбоорду ишке ашыруу",
        "материалдарды даярдоо",
        "чечимди кабыл алуу",
    ]
    ky_subj_topics = [
        "жөнүндө",
        "боюнча",
        "маселеси боюнча",
        "алкагында",
        "жыйынтыктары жөнүндө",
    ]
    ky_subjects = []
    for v in ky_subj_verbs:
        for t in ky_subj_topics:
            ky_subjects.append(f"{v} {t}")
    for s in _DATA["ky"]["subjects"]:
        ky_subjects.append(s)
    local_rng.shuffle(ky_subjects)
    _DATA["ky"]["subjects"] = ky_subjects[:300]

    ky_recipients = []
    for n in _DATA["ky"]["names"][:50]:
        parts = n.split()
        if len(parts) >= 2:
            first, last = parts[0], " ".join(parts[1:])
            ky_recipients.append(f"Башкы директор {first} {last} мырзага")
            ky_recipients.append(f"Бөлүм башчысы {first} {last}-га")
    _DATA["ky"]["recipient"] = ky_recipients[:100]


_populate_large_pools()


def _merge_typed_corpus() -> None:
    import unicodedata

    resource_map = {
        "organizations": "organizations_{language}.jsonl",
        "departments": "departments_{language}.jsonl",
        "names": "people_{language}.jsonl",
        "addresses": "addresses_{language}.jsonl",
        "subjects": "official_subjects_{language}.jsonl",
    }
    for language in ("kk", "ky"):
        for field, filename_template in resource_map.items():
            records = load_corpus_records(filename_template.format(language=language))
            existing = list(_DATA[language][field])
            known = set(existing)
            for record in records:
                if record.language == language and record.text not in known:
                    existing.append(record.text)
                    known.add(record.text)
            _DATA[language][field] = [
                unicodedata.normalize("NFC", str(x)) for x in existing
            ]


_merge_typed_corpus()
