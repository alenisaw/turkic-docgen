use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

pub fn string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field)?.as_str()
}

pub const REQUIRED_FIELDS: &[&str] = &[
    "id",
    "page_id",
    "image",
    "image_path",
    "layout_id",
    "language_mix",
    "quality_profile",
    "effect_metadata",
    "qa_ok",
    "qa_issues",
    "zones",
];

pub fn resolve_path_safe(
    base: &std::path::Path,
    value: &str,
) -> Result<std::path::PathBuf, String> {
    let raw = std::path::Path::new(value);
    if raw.is_absolute() {
        return Err(format!("absolute path not allowed: {value}"));
    }
    for component in raw.components() {
        match component {
            std::path::Component::ParentDir => {
                return Err(format!("path traversal ('..') not allowed: {value}"));
            }
            std::path::Component::Prefix(_) => {
                return Err(format!("path drive prefix not allowed: {value}"));
            }
            _ => {}
        }
    }
    Ok(base.join(raw))
}

pub fn read_manifest(path: &std::path::Path) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("cannot read manifest: {err}"))?;
    let mut rows = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("line {}: invalid JSON: {err}", line_idx + 1))?;
        rows.push(value);
    }
    Ok(rows)
}

pub fn increment_count(map: &mut BTreeMap<String, usize>, key: impl Into<String>) {
    *map.entry(key.into()).or_insert(0) += 1;
}

pub fn duplicate_count(rows: &[Value], field: &str) -> usize {
    let mut seen = HashSet::new();
    let mut duplicates = 0;
    for row in rows {
        if let Some(value) = string_field(row, field) {
            if !seen.insert(value.to_string()) {
                duplicates += 1;
            }
        }
    }
    duplicates
}

pub fn value_after(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let pos = args
        .iter()
        .position(|arg| arg == flag)
        .ok_or_else(|| format!("missing {flag}"))?;
    args.get(pos + 1)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing value for {flag}"))
}

pub fn string_after(args: &[String], flag: &str) -> Result<String, String> {
    let pos = args
        .iter()
        .position(|arg| arg == flag)
        .ok_or_else(|| format!("missing {flag}"))?;
    args.get(pos + 1)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

pub fn optional_string_after(args: &[String], flag: &str) -> Option<String> {
    let pos = args.iter().position(|arg| arg == flag)?;
    args.get(pos + 1).cloned()
}

pub fn zone_text(row: &Value) -> String {
    row.get("zones")
        .and_then(Value::as_array)
        .map(|zones| {
            zones
                .iter()
                .filter_map(|zone| zone.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}
