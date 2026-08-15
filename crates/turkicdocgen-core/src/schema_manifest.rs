use crate::hashing::sha256_file;
use crate::util::{read_manifest, string_field, REQUIRED_FIELDS};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

fn as_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_u64().and_then(|num| i64::try_from(num).ok()))
            .or_else(|| v.as_str().and_then(|text| text.parse::<i64>().ok()))
    })
}

fn resolve(root: &std::path::Path, value: &str) -> Result<PathBuf, String> {
    crate::util::resolve_path_safe(root, value)
}

pub fn schema_manifest_check(
    manifest: PathBuf,
    images_root: Option<PathBuf>,
) -> Result<(), String> {
    let rows = read_manifest(&manifest)?;
    let root = images_root.as_ref();
    let base = manifest
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut errors = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut seen_images = HashSet::new();

    for (idx, row) in rows.iter().enumerate() {
        let Some(obj) = row.as_object() else {
            errors.push(format!("row:{idx}:not_object"));
            continue;
        };
        let mut missing = REQUIRED_FIELDS
            .iter()
            .filter(|key| !obj.contains_key(**key))
            .copied()
            .collect::<Vec<_>>();
        missing.sort();
        if !missing.is_empty() {
            errors.push(format!("row:{idx}:missing:{}", missing.join(",")));
        }

        let sample_id = string_field(row, "id").unwrap_or("");
        if sample_id.is_empty() {
            errors.push(format!("row:{idx}:empty_id"));
        } else if !seen_ids.insert(sample_id.to_string()) {
            errors.push(format!("row:{idx}:duplicate_id:{sample_id}"));
        }
        let image = string_field(row, "image").unwrap_or("");
        if image.is_empty() {
            errors.push(format!("row:{idx}:empty_image"));
        } else if !seen_images.insert(image.to_string()) {
            errors.push(format!("row:{idx}:duplicate_image:{image}"));
        }
        for field in ["page_id", "layout_id", "language_mix", "quality_profile"] {
            if string_field(row, field).unwrap_or("").trim().is_empty() {
                errors.push(format!("row:{idx}:empty_{field}"));
            }
        }
        if row.get("qa_ok").and_then(Value::as_bool) != Some(true) {
            errors.push(format!("row:{idx}:qa_not_ok"));
        }

        let zones = match row.get("zones").and_then(Value::as_array) {
            Some(items) if !items.is_empty() => items,
            _ => {
                errors.push(format!("row:{idx}:zones_empty_or_invalid"));
                continue;
            }
        };
        for (zone_index, zone) in zones.iter().enumerate() {
            let Some(zone_obj) = zone.as_object() else {
                errors.push(format!("row:{idx}:zone:{zone_index}:not_object"));
                continue;
            };
            if zone_obj
                .get("zone_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .is_empty()
            {
                errors.push(format!("row:{idx}:zone:{zone_index}:empty_zone_id"));
            }
            if zone_obj
                .get("zone_type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .is_empty()
            {
                errors.push(format!("row:{idx}:zone:{zone_index}:empty_zone_type"));
            }
            let Some(bbox) = zone_obj.get("bbox").and_then(Value::as_array) else {
                errors.push(format!("row:{idx}:zone:{zone_index}:invalid_bbox"));
                continue;
            };
            if bbox.len() != 4 || bbox.iter().any(|value| as_i64(Some(value)).is_none()) {
                errors.push(format!("row:{idx}:zone:{zone_index}:invalid_bbox"));
                continue;
            }
            let left = as_i64(bbox.first()).unwrap();
            let top = as_i64(bbox.get(1)).unwrap();
            let right = as_i64(bbox.get(2)).unwrap();
            let bottom = as_i64(bbox.get(3)).unwrap();
            if right <= left || bottom <= top {
                errors.push(format!("row:{idx}:zone:{zone_index}:invalid_bbox"));
            }
        }

        let root = root.map(std::path::PathBuf::as_path).unwrap_or(base);
        let image_path = resolve(root, image)?;
        if !image.is_empty() && !image_path.exists() {
            errors.push(format!("row:{idx}:image_missing:{image}"));
        } else if let Some(expected) = string_field(row, "image_hash") {
            if !expected.is_empty()
                && sha256_file(&image_path)
                    .map(|actual| actual != expected)
                    .unwrap_or(false)
            {
                errors.push(format!("row:{idx}:image_hash_mismatch"));
            }
        }
    }

    if rows.is_empty() {
        errors.push("manifest_empty".to_string());
    }
    if errors.is_empty() {
        println!("schema manifest ok: {} rows", rows.len());
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}
