use crate::hashing::sha256_file;
use crate::util::{read_manifest, string_field, REQUIRED_FIELDS};
use std::path::PathBuf;

fn resolve_artifact(
    root: Option<&std::path::Path>,
    manifest: &std::path::Path,
    value: &str,
) -> Result<PathBuf, String> {
    if let Some(root_path) = root {
        crate::util::resolve_path_safe(root_path, value)
    } else {
        let parent = manifest
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        crate::util::resolve_path_safe(parent, value)
    }
}

pub fn manifest_check(path: PathBuf, images_root: Option<PathBuf>) -> Result<(), String> {
    let rows = read_manifest(&path)?;
    let mut errors: Vec<String> = Vec::new();

    for (idx, value) in rows.iter().enumerate() {
        let line_no = idx + 1;
        for field in REQUIRED_FIELDS {
            if value.get(*field).is_none() {
                errors.push(format!("line {line_no}: missing field {field}"));
            }
        }
        for field in [
            "id",
            "page_id",
            "image",
            "layout_id",
            "language_mix",
            "quality_profile",
        ] {
            if value.get(field).is_some() {
                match string_field(value, field) {
                    Some(text) if !text.trim().is_empty() => {}
                    Some(_) => errors.push(format!("line {line_no}: empty {field}")),
                    None => errors.push(format!("line {line_no}: {field} must be a string")),
                }
            }
        }
        if value.get("qa_ok").and_then(serde_json::Value::as_bool) != Some(true) {
            errors.push(format!("line {line_no}: qa_ok must be true"));
        }
        match value.get("zones").and_then(serde_json::Value::as_array) {
            Some(items) if !items.is_empty() => {}
            _ => errors.push(format!("line {line_no}: zones must be a non-empty array")),
        }
        if images_root.is_some() {
            if let Some(file_name) = string_field(value, "image") {
                let image_path = resolve_artifact(images_root.as_deref(), &path, file_name)?;
                if !image_path.exists() {
                    errors.push(format!(
                        "line {line_no}: image does not exist: {}",
                        image_path.to_string_lossy()
                    ));
                } else if let Some(expected) = string_field(value, "image_hash") {
                    let actual = sha256_file(&image_path)?;
                    if actual != expected {
                        errors.push(format!("line {line_no}: image_hash mismatch"));
                    }
                }
            }
        }
    }

    if rows.is_empty() {
        errors.push("manifest has no rows".to_string());
    }
    if errors.is_empty() {
        println!("manifest ok: {} rows", rows.len());
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}
