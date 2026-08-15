from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from turkicdocgen.profiles import dataset_family
from turkicdocgen.safety import (
    ROOT,
    assert_generated_path,
    resolve_generated_run,
)


class GalleryFilters(BaseModel):
    run_id: str | None = None
    output_base: str = "outputs"
    status_filter: str = "accepted"
    lang_filter: str = ""
    profile_filter: str = ""
    effect_filter: str = ""
    stamp_filter: str = ""
    warning_filter: str = ""
    domain_filter: str = ""
    page: int = 1
    page_size: int = 24

    def model_post_init(self, __context: Any) -> None:
        self.page = max(1, int(self.page or 1))
        self.page_size = _clamp_page_size(self.page_size)


GALLERY_PAGE_SIZES = (12, 24, 48, 96)
DEFAULT_GALLERY_PAGE_SIZE = 24
MAX_GALLERY_PAGE_SIZE = 96
STREAM_FALLBACK_SCAN_LIMIT = 2_000
RUN_DETAIL_PREVIEW_LIMIT = 50
REJECTED_PAGE_SIZE = 24
REJECTED_FALLBACK_SCAN_LIMIT = 2_000
MAX_FALLBACK_MANIFEST_ROWS = 100_000
_INDEX_BUILDING: set[str] = set()
_INDEX_LOCK = threading.Lock()


def _clamp_page_size(value: int | str | None) -> int:
    try:
        size = int(value or DEFAULT_GALLERY_PAGE_SIZE)
    except (TypeError, ValueError):
        return DEFAULT_GALLERY_PAGE_SIZE
    if size in GALLERY_PAGE_SIZES:
        return size
    if size > MAX_GALLERY_PAGE_SIZE:
        return MAX_GALLERY_PAGE_SIZE
    return DEFAULT_GALLERY_PAGE_SIZE


_HERE = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

LOCALIZATION = {
    "ru": {
        "dashboard": "Галерея",
        "generate": "Генерация",
        "runs": "Запуски",
        "output_runs": "Запуски",
        "total_jobs": "Задачи",
        "running": "В работе",
        "completed": "Готово",
        "quick_generate": "Быстрая генерация",
        "profile": "Профиль",
        "count": "Количество",
        "output_dir": "Папка вывода",
        "seed": "Seed",
        "start_generation": "Запустить",
        "active_jobs": "Задачи генерации",
        "refresh": "Обновить",
        "no_jobs_yet": "Задач генерации нет.",
        "recent_runs": "Последние запуски",
        "view_all": "Все",
        "run_id": "Запуск",
        "samples": "док.",
        "actions": "Действия",
        "detail": "Детали",
        "gallery": "Галерея",
        "no_runs_found": "Запуски не найдены.",
        "rejected": "Отклонено",
        "logs": "Логи",
        "cancel": "Отменить",
        "status": "Статус",
        "all_runs": "Все запуски",
        "path": "Путь",
        "delete": "Удалить",
        "confirm_delete": "Удалить выбранный запуск?",
        "delete_run_title": "Удаление запуска",
        "soft_delete": "Очистить медиа",
        "soft_delete_desc": "Удалить изображения, зоны и отчеты, сохранив manifest.",
        "hard_delete": "Удалить полностью",
        "hard_delete_desc": "Удалить весь каталог запуска без восстановления.",
        "cancel_btn": "Отмена",
        "run_details": "Детали запуска",
        "overview": "Сводка",
        "qa_accepted": "QA принято",
        "qa_rejected": "QA отклонено",
        "near_duplicates": "Похожие",
        "repeated_skeletons": "Повторы",
        "manifest_records": "Manifest",
        "no_records_in_manifest": "Manifest пуст.",
        "start_new_job": "Новая генерация",
        "job_settings": "Параметры",
        "num_workers": "Потоки",
        "auto_workers": "Авто",
        "back_to_dashboard": "К галерее",
        "visual_gallery": "Визуальная QA-галерея",
        "all": "Все",
        "accepted": "Принято",
        "language": "Язык",
        "page": "Страница",
        "prev": "Назад",
        "next": "Вперед",
        "lang_filter_placeholder": "Язык",
        "profile_filter_placeholder": "Layout",
        "effect_filter_placeholder": "Эффект",
        "apply_filter": "Фильтр",
        "no_samples_match": "Нет документов по фильтрам",
        "clear_filters": "Сбросить",
        "rejected_samples": "Отклоненные",
        "reason_filter": "Причина",
        "no_rejected_samples": "Отклоненных документов нет.",
        "sample_detail_title": "Документ",
        "qa_status": "QA",
        "language_policy": "Язык",
        "license": "Лицензия",
        "pii_status": "PII",
        "visual_review_status": "Ручная проверка",
        "reviewer_note": "Комментарий проверяющего",
        "save_review": "Сохранить",
        "zone_ground_truth": "Зоны",
        "text_content": "Текст",
        "bbox": "BBox",
        "back_to_gallery": "В галерею",
        "back_to_run": "К запуску",
        "no_image_available": "Изображение недоступно",
        "tdg_title": "TurkicDocGen QA",
        "success": "Готово",
        "error": "Ошибка",
        "soft_deleted_status": "медиа очищены",
        "export": "Экспорт",
        "export_success": "Экспорт выполнен.",
        "export_failed": "Экспорт не выполнен.",
        "exporting": "Экспорт...",
        "accept": "Принять",
        "reject": "Отклонить",
        "flag": "Пометить",
        "reset": "Сбросить",
        "stamp": "Штамп",
        "qa_warnings": "Предупреждения",
        "corpus_domain": "Домен корпуса",
        "flagged": "Помечено",
        "pending_manual": "Ожидает проверки",
        "image": "Изображение",
        "hide_zones": "Скрыть зоны",
        "show_zones": "Показать зоны",
        "open": "Открыть",
        "metadata": "Метаданные",
        "effect_chain": "Цепочка эффектов",
        "regenerate": "Регенерация",
        "zones_count": "зон",
        "accept_rate": "Доля прошедших контроль",
        "browse_images_desc": "Просмотр сгенерированных изображений",
        "view_rejected_desc": "Просмотр отбракованных документов",
        "delete_purge_desc": "Удалить или очистить тяжелые файлы",
        "rows_count": "строк",
        "showing_50_of": "Показано 50 из {total} строк",
        "all_reasons": "Все причины",
        "shown": "показано",
        "no_rejected_samples_title": "Нет отбракованных документов",
        "no_image": "Нет изображения",
        "yaml_config_hint": "Конфигурация YAML в configs/generator/",
        "output_dir_hint": "Относительно корня проекта. Будет создана при отсутствии.",
        "workers_hint": "Число параллельных потоков. Используйте 1 для детерминированных запусков.",
        "start_gen_logs_hint": "Запустите генерацию, чтобы увидеть журнал работы в реальном времени.",
        "layout": "Макет",
        "qa": "QA",
        "qa_ok": "QA ОК",
        "qa_fail": "QA ОШИБКА",
        "all_abbr": "ВСЕ",
        "ok_abbr": "ОК",
        "no_abbr": "НЕТ",
        "qa_abbr": "QA",
        "img_abbr": "ИЗБ",
        "del_abbr": "УДЛ",
        "yes": "Есть",
        "no_val": "Нет",
        "auto": "авто",
        "command": "Команда",
        "zoom": "Масштаб",
        "fonts": "Шрифты",
        "raw": "Исходные",
        "inspector_layers": "Слои инспектора",
        "layer_zones": "Зоны",
        "layer_lines": "Строки",
        "layer_cells": "Ячейки таблиц",
        "layer_reading_order": "Порядок чтения",
        "layer_qa_issues": "Проблемы QA",
        "layer_decorations": "Декор и штампы",
        "label_mode": "Режим подписей",
        "label_none": "Скрыть",
        "label_number": "Номер",
        "label_role": "Роль",
        "label_id": "Zone ID",
        "filter_role": "Фильтр ролей",
        "all_roles": "Все роли",
        "opacity": "Прозрачность",
        "show_only_problems": "Только проблемы QA",
        "reset_view": "Сбросить вид",
        "fit_page": "Вписать страницу",
        "fit_width": "Вписать по ширине",
    },
    "en": {
        "dashboard": "Gallery",
        "generate": "Generate",
        "runs": "Runs",
        "output_runs": "Runs",
        "total_jobs": "Jobs",
        "running": "Running",
        "completed": "Done",
        "quick_generate": "Quick generate",
        "profile": "Profile",
        "count": "Count",
        "output_dir": "Output directory",
        "seed": "Seed",
        "start_generation": "Start",
        "active_jobs": "Generation jobs",
        "refresh": "Refresh",
        "no_jobs_yet": "No generation jobs.",
        "recent_runs": "Recent runs",
        "view_all": "All",
        "run_id": "Run",
        "samples": "samples",
        "actions": "Actions",
        "detail": "Details",
        "gallery": "Gallery",
        "no_runs_found": "No runs found.",
        "rejected": "Rejected",
        "logs": "Logs",
        "cancel": "Cancel",
        "status": "Status",
        "all_runs": "All runs",
        "path": "Path",
        "delete": "Delete",
        "confirm_delete": "Delete the selected run?",
        "delete_run_title": "Delete run",
        "soft_delete": "Clean media",
        "soft_delete_desc": "Delete images, zones, and reports while keeping manifest.",
        "hard_delete": "Delete fully",
        "hard_delete_desc": "Delete the run directory permanently.",
        "cancel_btn": "Cancel",
        "run_details": "Delete run",
        "overview": "Overview",
        "qa_accepted": "QA accepted",
        "qa_rejected": "QA rejected",
        "near_duplicates": "Near duplicates",
        "repeated_skeletons": "Repeated skeletons",
        "manifest_records": "Manifest",
        "no_records_in_manifest": "Manifest is empty.",
        "start_new_job": "New generation",
        "job_settings": "Settings",
        "num_workers": "Workers",
        "auto_workers": "Auto",
        "back_to_dashboard": "Back to gallery",
        "visual_gallery": "Visual QA gallery",
        "all": "All",
        "accepted": "Accepted",
        "language": "Language",
        "page": "Page",
        "prev": "Prev",
        "next": "Next",
        "lang_filter_placeholder": "Language",
        "profile_filter_placeholder": "Layout",
        "effect_filter_placeholder": "Effect",
        "apply_filter": "Filter",
        "no_samples_match": "No matching documents",
        "clear_filters": "Clear",
        "rejected_samples": "Rejected",
        "reason_filter": "Reason",
        "no_rejected_samples": "No rejected documents.",
        "sample_detail_title": "Document",
        "qa_status": "QA",
        "language_policy": "Language",
        "license": "License",
        "pii_status": "PII",
        "visual_review_status": "Manual review",
        "reviewer_note": "Reviewer note",
        "save_review": "Save",
        "zone_ground_truth": "Zones",
        "text_content": "Text",
        "bbox": "BBox",
        "back_to_gallery": "Back to gallery",
        "back_to_run": "Back to run",
        "no_image_available": "Image unavailable",
        "tdg_title": "TurkicDocGen QA",
        "success": "Done",
        "error": "Error",
        "soft_deleted_status": "media cleaned",
        "export": "Export",
        "export_success": "Export completed.",
        "export_failed": "Export failed.",
        "exporting": "Exporting...",
        "accept": "Accept",
        "reject": "Reject",
        "flag": "Flag",
        "reset": "Reset",
        "stamp": "Stamp",
        "qa_warnings": "Warnings",
        "corpus_domain": "Corpus domain",
        "flagged": "Flagged",
        "pending_manual": "Pending review",
        "image": "Image",
        "hide_zones": "Hide zones",
        "show_zones": "Show zones",
        "open": "Open",
        "metadata": "Metadata",
        "effect_chain": "Effect chain",
        "regenerate": "Regenerate",
        "zones_count": "zones",
        "accept_rate": "Accept Rate",
        "browse_images_desc": "Browse generated images",
        "view_rejected_desc": "View rejected samples",
        "delete_purge_desc": "Remove or purge media files",
        "rows_count": "rows",
        "showing_50_of": "Showing 50 of {total} rows",
        "all_reasons": "All reasons",
        "shown": "shown",
        "no_rejected_samples_title": "No rejected samples",
        "no_image": "No image",
        "yaml_config_hint": "YAML config in configs/generator/",
        "output_dir_hint": "Relative to project root. Will be created if missing.",
        "workers_hint": "Parallel generation workers. Use 1 for deterministic single-seed runs.",
        "start_gen_logs_hint": "Start a generation to see live logs here.",
        "layout": "Layout",
        "qa": "QA",
        "qa_ok": "QA OK",
        "qa_fail": "QA FAIL",
        "all_abbr": "ALL",
        "ok_abbr": "OK",
        "no_abbr": "NO",
        "qa_abbr": "QA",
        "img_abbr": "IMG",
        "del_abbr": "DEL",
        "yes": "Yes",
        "no_val": "No",
        "auto": "auto",
        "command": "Command",
        "zoom": "Zoom",
        "fonts": "Fonts",
        "raw": "Raw",
        "inspector_layers": "Inspector Layers",
        "layer_zones": "Zones",
        "layer_lines": "Lines",
        "layer_cells": "Table Cells",
        "layer_reading_order": "Reading Order",
        "layer_qa_issues": "QA Issues",
        "layer_decorations": "Decorations & Stamps",
        "label_mode": "Label Mode",
        "label_none": "None",
        "label_number": "Number",
        "label_role": "Role",
        "label_id": "Zone ID",
        "filter_role": "Filter Roles",
        "all_roles": "All Roles",
        "opacity": "Opacity",
        "show_only_problems": "Show Only Problems",
        "reset_view": "Reset View",
        "fit_page": "Fit Page",
        "fit_width": "Fit Width",
    },
}


def t(request: Request, key: str) -> str:
    lang = request.query_params.get("lang")
    if lang not in ("en", "ru"):
        lang = request.cookies.get("lang", "ru")
    return LOCALIZATION.get(lang, LOCALIZATION["ru"]).get(key, key)


def get_lang(request: Request) -> str:
    lang = request.query_params.get("lang")
    if lang not in ("en", "ru"):
        lang = request.cookies.get("lang", "ru")
    return lang


def get_theme(request: Request) -> str:
    theme = request.cookies.get("tdg-theme", "light")
    if theme not in ("light", "dark"):
        theme = "light"
    return theme


templates.env.globals["t"] = t
templates.env.globals["get_lang"] = get_lang
templates.env.globals["get_theme"] = get_theme


def _available_profiles() -> list[str]:
    """Return active generation profiles."""
    return sorted(dataset_family())


ALLOWED_OUTPUT_BASES = {"outputs", "release", "reports", "runs", "artifacts"}


def _output_base(output_base: str | Path = "outputs") -> Path:
    base_str = str(output_base).strip()
    if base_str not in ALLOWED_OUTPUT_BASES:
        raise HTTPException(
            status_code=403, detail=f"Forbidden output_base: {base_str}"
        )

    env_base = os.environ.get("TURKICDOCGEN_WEB_INPUT")
    target_str = env_base if env_base and base_str == "outputs" else base_str
    target_path = Path(target_str)

    try:
        parent_name = target_path.parent.name
        if (
            (target_path / "manifest.jsonl").exists()
            or (target_path.resolve() / "manifest.jsonl").exists()
            or parent_name in ALLOWED_OUTPUT_BASES
        ):
            target_path = target_path.parent
    except (OSError, RuntimeError, ValueError):
        pass

    try:
        return assert_generated_path(target_path, purpose="web output access")
    except ValueError:
        default_path = ROOT / target_str
        try:
            default_parent_name = default_path.parent.name
            if (
                (default_path / "manifest.jsonl").exists()
                or (default_path.resolve() / "manifest.jsonl").exists()
                or default_parent_name in ALLOWED_OUTPUT_BASES
            ):
                default_path = default_path.parent
        except (OSError, RuntimeError, ValueError):
            pass
        return assert_generated_path(default_path, purpose="web output access")


def _run_dir(output_base: str | Path, run_id: str) -> Path:
    try:
        safe_base = _output_base(output_base)
        return resolve_generated_run(safe_base, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _list_runs(output_base: str = "outputs") -> list[dict[str, Any]]:
    base = _output_base(output_base)
    runs = []
    if base.exists():
        for subdir in sorted(
            base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if subdir.is_dir():
                manifest = subdir / "manifest.jsonl"
                summary = _read_run_summary(subdir)
                count = int(summary.get("count") or summary.get("accepted") or 0)
                if manifest.exists():
                    count = count or _bounded_manifest_count(manifest)
                is_soft_deleted = (subdir / ".soft_deleted").exists()
                accepted_estimate = int(summary.get("accepted") or 0)
                languages = sorted((summary.get("languages") or {}).keys())
                layouts = sorted((summary.get("layouts") or {}).keys())
                effects = sorted((summary.get("effects") or {}).keys())
                runs.append(
                    {
                        "run_id": subdir.name,
                        "path": str(subdir),
                        "sample_count": count,
                        "has_manifest": manifest.exists(),
                        "is_soft_deleted": is_soft_deleted,
                        "modified_at": subdir.stat().st_mtime,
                        "accepted": accepted_estimate,
                        "qa_rate": (
                            round(accepted_estimate / count * 100, 1) if count else 0.0
                        ),
                        "languages": languages,
                        "layouts": layouts,
                        "effects": effects,
                        "status": "cleaned"
                        if is_soft_deleted
                        else ("ready" if manifest.exists() else "incomplete"),
                    }
                )
    return runs


def _read_run_summary(run_dir: Path) -> dict[str, Any]:
    for path in (
        run_dir / "reports" / "generation_summary.json",
        run_dir / "run_manifest.json",
    ):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "counts" in payload and isinstance(payload["counts"], dict):
            counts = payload["counts"]
            return {
                "count": counts.get("accepted", 0) + counts.get("rejected", 0),
                "accepted": counts.get("accepted", 0),
            }
        return payload if isinstance(payload, dict) else {}
    return {}


def _bounded_manifest_count(manifest: Path, limit: int = 1000) -> int:
    count = 0
    try:
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
                    if count >= limit:
                        return count
    except OSError:
        return 0
    return count


def _read_manifest(run_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    manifest = run_dir / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    if not manifest.exists():
        return rows
    with open(manifest, encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            if i >= limit:
                break
            line = raw_line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(_normalize_manifest_row(json.loads(line)))
    return rows


def _manifest_index_path(run_dir: Path) -> Path:
    return run_dir / ".web_manifest_index.sqlite3"


def _manifest_signature(manifest: Path) -> tuple[int, int]:
    stat = manifest.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _indexed_row_values(index: int, row: dict[str, Any]) -> tuple[Any, ...]:
    page_id = str(row.get("id") or row.get("page_id") or row.get("sample_id") or "")
    effect = str(row.get("effect_profile") or row.get("quality_profile") or "")
    quality = str(row.get("quality_profile") or "")
    warnings = _flatten_search_text(
        [
            row.get("qa_flags"),
            row.get("qa_issues"),
            _effect_metadata(row).get("warnings"),
        ]
    )
    domain = _flatten_search_text(_corpus_metadata(row))
    return (
        index,
        page_id,
        str(row.get("image") or ""),
        1 if row.get("qa_ok") is True else 0,
        str(row.get("language_mix") or ""),
        str(row.get("layout_id") or ""),
        effect,
        quality,
        str(row.get("orientation") or ""),
        1 if _has_stamp(row) else 0,
        warnings,
        domain,
    )


def _manifest_index_ready(run_dir: Path) -> bool:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        return False
    index_path = _manifest_index_path(run_dir)
    if not index_path.exists():
        return False
    manifest_mtime_ns, manifest_size = _manifest_signature(manifest)
    try:
        with sqlite3.connect(index_path) as db:
            current = dict(db.execute("SELECT key, value FROM metadata").fetchall())
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(rows)").fetchall()
            }
    except sqlite3.Error:
        return False
    required = {
        "row_index",
        "sample_id",
        "image",
        "qa_ok",
        "language_mix",
        "layout_id",
        "effect_profile",
        "quality_profile",
        "orientation",
        "has_stamp",
        "warning_text",
        "domain_text",
    }
    return (
        current.get("manifest_mtime_ns") == str(manifest_mtime_ns)
        and current.get("manifest_size") == str(manifest_size)
        and required.issubset(columns)
    )


def _ensure_manifest_index(run_dir: Path) -> Path | None:
    """Return a ready index path without building it."""
    index_path = _manifest_index_path(run_dir)
    return index_path if _manifest_index_ready(run_dir) else None


def _index_build_key(run_dir: Path) -> str:
    return str(run_dir.resolve())


def _manifest_index_status(run_dir: Path) -> dict[str, Any]:
    manifest = run_dir / "manifest.jsonl"
    key = _index_build_key(run_dir)
    with _INDEX_LOCK:
        building = key in _INDEX_BUILDING
    if not manifest.exists():
        return {
            "state": "missing_manifest",
            "ready": False,
            "message": "Manifest missing.",
        }
    if building:
        return {
            "state": "building",
            "ready": False,
            "message": "Manifest index is building; gallery is using bounded fallback.",
        }
    if _manifest_index_ready(run_dir):
        index_path = _manifest_index_path(run_dir)
        try:
            with sqlite3.connect(index_path) as db:
                rows = int(db.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        except sqlite3.Error:
            rows = None
        return {
            "state": "ready",
            "ready": True,
            "rows": rows,
            "message": "Manifest index ready.",
        }
    if _manifest_index_path(run_dir).exists():
        return {
            "state": "stale",
            "ready": False,
            "message": "Manifest index is stale; gallery is using bounded fallback.",
        }
    return {
        "state": "missing",
        "ready": False,
        "message": "Manifest index missing; gallery is using bounded fallback.",
    }


def _build_manifest_index(run_dir: Path) -> Path | None:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        return None
    index_path = _manifest_index_path(run_dir)
    manifest_mtime_ns, manifest_size = _manifest_signature(manifest)
    key = _index_build_key(run_dir)
    with _INDEX_LOCK:
        _INDEX_BUILDING.add(key)
    tmp_path = index_path.with_suffix(".sqlite3.tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        db = sqlite3.connect(tmp_path)
        try:
            db.execute("PRAGMA journal_mode = OFF")
            db.execute("PRAGMA synchronous = OFF")
            db.execute(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            db.executescript(
                """
                CREATE TABLE rows (
                    row_index INTEGER PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    image TEXT NOT NULL,
                    qa_ok INTEGER NOT NULL,
                    language_mix TEXT NOT NULL,
                    layout_id TEXT NOT NULL,
                    effect_profile TEXT NOT NULL,
                    quality_profile TEXT NOT NULL,
                    orientation TEXT NOT NULL,
                    has_stamp INTEGER NOT NULL,
                    warning_text TEXT NOT NULL,
                    domain_text TEXT NOT NULL
                );
                CREATE INDEX rows_sample_id ON rows(sample_id);
                CREATE INDEX rows_qa_ok ON rows(qa_ok);
                CREATE INDEX rows_language_mix ON rows(language_mix);
                CREATE INDEX rows_layout_id ON rows(layout_id);
                CREATE INDEX rows_effect_profile ON rows(effect_profile);
                CREATE INDEX rows_has_stamp ON rows(has_stamp);
                """
            )
            batch: list[tuple[Any, ...]] = []
            with manifest.open(encoding="utf-8") as fh:
                for index, raw_line in enumerate(fh):
                    line = raw_line.strip()
                    if not line:
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        row = _normalize_manifest_row(json.loads(line))
                        batch.append(_indexed_row_values(index, row))
                    if len(batch) >= 1000:
                        db.executemany(
                            """
                            INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            batch,
                        )
                        batch.clear()
                if batch:
                    db.executemany(
                        "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
            db.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("manifest_mtime_ns", str(manifest_mtime_ns)),
                    ("manifest_size", str(manifest_size)),
                    ("built_at", str(time.time())),
                ],
            )
            db.commit()
        finally:
            db.close()
        tmp_path.replace(index_path)
        return index_path
    except (OSError, sqlite3.Error):
        tmp_path.unlink(missing_ok=True)
        return None
    finally:
        with _INDEX_LOCK:
            _INDEX_BUILDING.discard(key)


def _gallery_sql(filters: GalleryFilters) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if filters.status_filter and filters.status_filter != "all":
        clauses.append("qa_ok = ?")
        params.append(1 if filters.status_filter == "accepted" else 0)
    if filters.lang_filter:
        clauses.append("language_mix LIKE ?")
        params.append(f"%{filters.lang_filter}%")
    if filters.profile_filter:
        clauses.append("layout_id LIKE ?")
        params.append(f"%{filters.profile_filter}%")
    if filters.effect_filter:
        clauses.append("effect_profile LIKE ?")
        params.append(f"%{filters.effect_filter}%")
    if filters.stamp_filter:
        if filters.stamp_filter == "with_stamp":
            clauses.append("has_stamp = 1")
        elif filters.stamp_filter == "without_stamp":
            clauses.append("has_stamp = 0")
        else:
            clauses.append("has_stamp = 1")
    if filters.warning_filter:
        clauses.append("warning_text LIKE ?")
        params.append(f"%{filters.warning_filter}%")
    if filters.domain_filter:
        clauses.append("domain_text LIKE ?")
        params.append(f"%{filters.domain_filter}%")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _read_indexed_gallery(
    run_dir: Path,
    filters: GalleryFilters,
) -> tuple[list[dict[str, Any]], int, dict[str, list[str]]]:
    index_path = _ensure_manifest_index(run_dir)
    if index_path is None:
        return _stream_gallery_page(run_dir, filters)
    where, params = _gallery_sql(filters)
    offset = (filters.page - 1) * filters.page_size
    with sqlite3.connect(index_path) as db:
        total = int(
            db.execute(f"SELECT COUNT(*) FROM rows{where}", params).fetchone()[0]
        )
        rows = [
            _gallery_row_from_index(row)
            for row in db.execute(
                f"""
                SELECT
                    row_index, sample_id, image, qa_ok, language_mix, layout_id,
                    effect_profile, quality_profile, orientation, has_stamp,
                    warning_text, domain_text
                FROM rows{where}
                ORDER BY row_index
                LIMIT ? OFFSET ?
                """,
                [*params, filters.page_size, offset],
            )
        ]
        options = {
            "unique_languages": _distinct_index_values(db, "language_mix"),
            "unique_profiles": _distinct_index_values(db, "layout_id"),
            "unique_effects": _distinct_index_values(db, "effect_profile"),
            "unique_warnings": _distinct_index_values(db, "warning_text"),
            "unique_domains": _distinct_index_values(db, "domain_text"),
        }
    return rows, total, options


def _gallery_row_from_index(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    (
        row_index,
        sample_id,
        image,
        qa_ok,
        language_mix,
        layout_id,
        effect_profile,
        quality_profile,
        orientation,
        has_stamp,
        warning_text,
        domain_text,
    ) = row
    warnings = [part for part in str(warning_text or "").split()[:2] if part]
    return {
        "row_index": int(row_index),
        "id": str(sample_id),
        "sample_id": str(sample_id),
        "image": str(image or ""),
        "qa_ok": bool(qa_ok),
        "language_mix": str(language_mix or ""),
        "layout_id": str(layout_id or ""),
        "effect_profile": str(effect_profile or ""),
        "quality_profile": str(quality_profile or ""),
        "orientation": str(orientation or ""),
        "has_stamp": bool(has_stamp),
        "qa_flags": warnings,
        "warning_text": str(warning_text or ""),
        "domain_text": str(domain_text or ""),
    }


def _stream_gallery_page(
    run_dir: Path,
    filters: GalleryFilters,
) -> tuple[list[dict[str, Any]], int, dict[str, list[str]]]:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        return [], 0, _extract_filter_options([])
    start = (filters.page - 1) * filters.page_size
    end = start + filters.page_size
    page_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    total = 0
    scanned = 0
    with manifest.open(encoding="utf-8") as fh:
        for raw_line in fh:
            scanned += 1
            if (
                scanned > STREAM_FALLBACK_SCAN_LIMIT
                and len(page_rows) >= filters.page_size
            ):
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = _normalize_manifest_row(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(option_rows) < 1000:
                option_rows.append(row)
            if not _filter_gallery_rows([row], filters):
                continue
            if start <= total < end:
                page_rows.append(row)
            total += 1
            if total >= end and scanned >= STREAM_FALLBACK_SCAN_LIMIT:
                break
    return page_rows, total, _extract_filter_options(option_rows)


def _distinct_index_values(db: sqlite3.Connection, column: str) -> list[str]:
    values = set()
    for (value,) in db.execute(
        f"SELECT DISTINCT {column} FROM rows WHERE {column} != ''"
    ):
        for part in str(value).split():
            if part:
                values.add(part)
    return sorted(values)


def _run_detail_data(run_dir: Path) -> dict[str, Any]:
    summary = _read_run_summary(run_dir)
    index_path = _ensure_manifest_index(run_dir)
    index_status = _manifest_index_status(run_dir)
    if index_path is not None:
        with sqlite3.connect(index_path) as db:
            total = int(db.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
            accepted = int(
                db.execute("SELECT COUNT(*) FROM rows WHERE qa_ok = 1").fetchone()[0]
            )
            preview = [
                _gallery_row_from_index(row)
                for row in db.execute(
                    """
                    SELECT
                        row_index, sample_id, image, qa_ok, language_mix, layout_id,
                        effect_profile, quality_profile, orientation, has_stamp,
                        warning_text, domain_text
                    FROM rows
                    ORDER BY row_index
                    LIMIT ?
                    """,
                    (RUN_DETAIL_PREVIEW_LIMIT,),
                )
            ]
        return {
            "rows": preview,
            "total": total,
            "accepted_count": accepted,
            "rejected_count": total - accepted,
            "bounded": False,
            "index_status": index_status,
        }
    if summary:
        total = int(summary.get("count") or 0)
        accepted = int(summary.get("accepted") or 0)
    else:
        total = 0
        accepted = 0
    rows = _read_manifest(run_dir, limit=RUN_DETAIL_PREVIEW_LIMIT)
    if not summary:
        total = len(rows)
        accepted = sum(1 for row in rows if row.get("qa_ok") is True)
    return {
        "rows": rows,
        "total": total,
        "accepted_count": accepted,
        "rejected_count": max(0, total - accepted),
        "bounded": True,
        "index_status": index_status,
    }


def _read_rejected_page(
    run_dir: Path,
    *,
    page: int = 1,
    page_size: int = REJECTED_PAGE_SIZE,
    reason_filter: str = "",
) -> dict[str, Any]:
    page = max(1, page)
    page_size = _clamp_page_size(page_size)
    index_path = _ensure_manifest_index(run_dir)
    if index_path is not None:
        where = " WHERE qa_ok = 0"
        params: list[Any] = []
        if reason_filter:
            where += " AND warning_text LIKE ?"
            params.append(f"%{reason_filter}%")
        offset = (page - 1) * page_size
        with sqlite3.connect(index_path) as db:
            total = int(
                db.execute(f"SELECT COUNT(*) FROM rows{where}", params).fetchone()[0]
            )
            rows = [
                _gallery_row_from_index(row)
                for row in db.execute(
                    f"""
                    SELECT
                        row_index, sample_id, image, qa_ok, language_mix, layout_id,
                        effect_profile, quality_profile, orientation, has_stamp,
                        warning_text, domain_text
                    FROM rows{where}
                    ORDER BY row_index
                    LIMIT ? OFFSET ?
                    """,
                    [*params, page_size, offset],
                )
            ]
            reason_types = _distinct_index_values(db, "warning_text")
        return {
            "rows": rows,
            "total": total,
            "reason_types": reason_types,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "bounded": False,
            "index_status": _manifest_index_status(run_dir),
        }
    return _stream_rejected_page(
        run_dir, page=page, page_size=page_size, reason_filter=reason_filter
    )


def _stream_rejected_page(
    run_dir: Path,
    *,
    page: int,
    page_size: int,
    reason_filter: str,
) -> dict[str, Any]:
    manifest = run_dir / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    reason_types: set[str] = set()
    total = 0
    start = (page - 1) * page_size
    end = start + page_size
    if manifest.exists():
        try:
            with manifest.open(encoding="utf-8") as fh:
                for scanned, raw_line in enumerate(fh, start=1):
                    if (
                        scanned > REJECTED_FALLBACK_SCAN_LIMIT
                        and len(rows) >= page_size
                    ):
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = _normalize_manifest_row(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if row.get("qa_ok") is True:
                        continue
                    for issue in row.get("qa_issues", []):
                        reason_types.add(
                            str(issue.get("code", issue))
                            if isinstance(issue, dict)
                            else str(issue)
                        )
                    if reason_filter and not _matches_warning(row, reason_filter):
                        continue
                    if start <= total < end:
                        rows.append(row)
                    total += 1
        except OSError:
            pass
    return {
        "rows": rows,
        "total": total,
        "reason_types": sorted(reason_types),
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "bounded": True,
        "index_status": _manifest_index_status(run_dir),
    }


def _find_indexed_manifest_row(
    run_dir: Path,
    sample_id: str,
) -> tuple[dict[str, Any], str | None, str | None, int, int] | None:
    index_path = _ensure_manifest_index(run_dir)
    if index_path is None:
        found = _find_manifest_row_streaming(
            run_dir, sample_id, MAX_FALLBACK_MANIFEST_ROWS
        )
        if found is None:
            return None
        row, prev_id, next_id, idx, total = found
        return (
            row,
            prev_id,
            next_id,
            idx,
            total,
        )
    with sqlite3.connect(index_path) as db:
        found = db.execute(
            "SELECT row_index FROM rows WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        if found is None:
            return None
        row_index = int(found[0])
        prev_row = db.execute(
            """
            SELECT sample_id FROM rows
            WHERE row_index < ?
            ORDER BY row_index DESC
            LIMIT 1
            """,
            (row_index,),
        ).fetchone()
        next_row = db.execute(
            """
            SELECT sample_id FROM rows
            WHERE row_index > ?
            ORDER BY row_index ASC
            LIMIT 1
            """,
            (row_index,),
        ).fetchone()
        total = int(db.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    row = _read_manifest_row_at_index(run_dir, row_index)
    if row is None:
        return None
    return (
        row,
        str(prev_row[0]) if prev_row else None,
        str(next_row[0]) if next_row else None,
        row_index,
        total,
    )


def _read_manifest_row_at_index(run_dir: Path, row_index: int) -> dict[str, Any] | None:
    manifest = run_dir / "manifest.jsonl"
    try:
        with manifest.open(encoding="utf-8") as fh:
            for index, raw_line in enumerate(fh):
                if index != row_index:
                    continue
                line = raw_line.strip()
                if not line:
                    return None
                return _normalize_manifest_row(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _find_manifest_row_streaming(
    run_dir: Path,
    sample_id: str,
    limit: int,
) -> tuple[dict[str, Any], str | None, str | None, int, int] | None:
    manifest = run_dir / "manifest.jsonl"
    prev_id: str | None = None
    found_row: dict[str, Any] | None = None
    found_prev: str | None = None
    found_index = -1
    total = 0
    try:
        with manifest.open(encoding="utf-8") as fh:
            for raw_index, raw_line in enumerate(fh):
                if raw_index >= limit:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = _normalize_manifest_row(json.loads(line))
                except json.JSONDecodeError:
                    continue
                current_id = str(row.get("id") or row.get("sample_id") or "")
                if found_row is not None:
                    return (
                        found_row,
                        found_prev,
                        current_id or None,
                        found_index,
                        total + 1,
                    )
                if current_id == sample_id:
                    found_row = row
                    found_prev = prev_id
                    found_index = raw_index
                prev_id = current_id or prev_id
                total += 1
    except OSError:
        return None
    if found_row is None:
        return None
    return found_row, found_prev, None, found_index, total


def _normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt old manifests for display without mutating generated output."""
    normalized = dict(row)
    normalized.setdefault("generator_schema_version", "legacy")
    normalized.setdefault("content_schema_id", None)
    normalized.setdefault("layout_variant", "legacy")
    normalized.setdefault("content_record_ids", [])
    effect_meta = normalized.get("effect_metadata")
    if not isinstance(effect_meta, dict):
        effect_meta = {}
    effect_meta.setdefault("effect_chain", normalized.get("effect_chain") or [])
    effect_meta.setdefault("warnings", [])
    effect_meta.setdefault("stamp_metadata", {})
    effect_meta.setdefault(
        "transform", {"kind": "legacy", "forward": None, "inverse": None}
    )
    normalized["effect_metadata"] = effect_meta
    groups = normalized.get("metadata_groups")
    if not isinstance(groups, dict):
        groups = {}
    defaults = {
        "identity": {"id": normalized.get("id"), "image": normalized.get("image")},
        "generation": {
            "generator_schema_version": normalized["generator_schema_version"]
        },
        "language": {"language_mix": normalized.get("language_mix")},
        "layout": {
            "layout_id": normalized.get("layout_id"),
            "content_schema_id": normalized.get("content_schema_id"),
            "layout_variant": normalized.get("layout_variant"),
        },
        "corpus": {"records": normalized.get("corpus_metadata") or []},
        "render": {"quality_profile": normalized.get("quality_profile")},
        "effects": effect_meta,
        "fonts": {"selected_fonts": normalized.get("selected_fonts") or []},
        "qa": {
            "qa_ok": normalized.get("qa_ok"),
            "qa_issues": normalized.get("qa_issues") or [],
            "qa_flags": normalized.get("qa_flags") or [],
        },
        "review": {
            "visual_qa_status": normalized.get("visual_qa_status", "pending_manual")
        },
        "release": {},
    }
    normalized["metadata_groups"] = {
        key: groups.get(key, value) for key, value in defaults.items()
    }
    normalized["has_stamp"] = _has_stamp(normalized)
    return normalized


def _flatten_search_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key} {_flatten_search_text(nested)}" for key, nested in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_search_text(item) for item in value)
    return str(value)


def _effect_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("effect_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _has_stamp(row: dict[str, Any]) -> bool:
    stamp = _effect_metadata(row).get("stamp_metadata")
    return isinstance(stamp, dict) and bool(
        stamp.get("stamp_id") or stamp.get("stamp_text")
    )


def _corpus_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("corpus_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _matches_status(row: dict[str, Any], value: str) -> bool:
    if value == "accepted":
        return row.get("qa_ok") is True
    if value == "rejected":
        return row.get("qa_ok") is not True
    return True


def _matches_text(value: Any, needle: str) -> bool:
    return needle.casefold() in str(value or "").casefold()


def _matches_stamp(row: dict[str, Any], value: str) -> bool:
    if value == "with_stamp":
        return _has_stamp(row)
    if value == "without_stamp":
        return not _has_stamp(row)
    stamp = _effect_metadata(row).get("stamp_metadata")
    return _matches_text(_flatten_search_text(stamp), value)


def _matches_warning(row: dict[str, Any], value: str) -> bool:
    warnings = [
        row.get("qa_flags"),
        row.get("qa_issues"),
        _effect_metadata(row).get("warnings"),
    ]
    return _matches_text(_flatten_search_text(warnings), value)


def _filter_gallery_rows(
    rows: list[dict[str, Any]],
    filters: GalleryFilters,
) -> list[dict[str, Any]]:
    status_filter = filters.status_filter
    lang_filter = filters.lang_filter
    profile_filter = filters.profile_filter
    effect_filter = filters.effect_filter
    stamp_filter = filters.stamp_filter
    warning_filter = filters.warning_filter
    domain_filter = filters.domain_filter
    predicates = []
    if status_filter and status_filter != "all":
        predicates.append(lambda row: _matches_status(row, status_filter))
    if lang_filter:
        predicates.append(
            lambda row: _matches_text(row.get("language_mix"), lang_filter)
        )
    if profile_filter:
        predicates.append(
            lambda row: _matches_text(row.get("layout_id"), profile_filter)
        )
    if effect_filter:
        predicates.append(
            lambda row: _matches_text(
                row.get("effect_profile") or row.get("quality_profile"),
                effect_filter,
            )
        )
    if stamp_filter:
        predicates.append(lambda row: _matches_stamp(row, stamp_filter))
    if warning_filter:
        predicates.append(lambda row: _matches_warning(row, warning_filter))
    if domain_filter:
        predicates.append(
            lambda row: _matches_text(
                _flatten_search_text(_corpus_metadata(row)), domain_filter
            )
        )
    return [row for row in rows if all(predicate(row) for predicate in predicates)]


def _warning_options(rows: list[dict[str, Any]]) -> list[str]:
    values = set()
    for row in rows:
        values.update(str(flag) for flag in row.get("qa_flags") or [])
        for issue in row.get("qa_issues") or []:
            values.add(
                str(issue.get("code", issue) if isinstance(issue, dict) else issue)
            )
        values.update(
            str(warning) for warning in (_effect_metadata(row).get("warnings") or [])
        )
    return sorted(values)


def _domain_options(rows: list[dict[str, Any]]) -> list[str]:
    values = set()
    for row in rows:
        corpus = row.get("corpus_metadata") or []
        records = corpus if isinstance(corpus, list) else [corpus]
        values.update(
            str(record["domain"])
            for record in records
            if isinstance(record, dict) and "domain" in record
        )
    return sorted(values)


def _extract_filter_options(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    unique_languages = sorted(
        {str(row.get("language_mix", "")) for row in rows if row.get("language_mix")}
    )
    unique_profiles = sorted(
        {str(row.get("layout_id", "")) for row in rows if row.get("layout_id")}
    )
    unique_effects = sorted(
        {
            str(row.get("effect_profile") or row.get("quality_profile") or "")
            for row in rows
            if row.get("effect_profile") or row.get("quality_profile")
        }
    )

    return {
        "unique_languages": unique_languages,
        "unique_profiles": unique_profiles,
        "unique_effects": unique_effects,
        "unique_warnings": _warning_options(rows),
        "unique_domains": _domain_options(rows),
    }
