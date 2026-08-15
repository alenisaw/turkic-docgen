use crate::util::{read_manifest, string_field};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

#[derive(Clone)]
struct IndexedRow {
    id: String,
    value: String,
}

fn key_set(rows: &[Value], field: &str) -> HashSet<String> {
    rows.iter()
        .filter_map(|row| string_field(row, field).map(ToOwned::to_owned))
        .collect()
}

fn indexed_rows(rows: &[Value], field: &str) -> Vec<IndexedRow> {
    rows.iter()
        .filter_map(|row| {
            let value = string_field(row, field)?;
            let id = string_field(row, "id").unwrap_or("").to_string();
            Some(IndexedRow {
                id,
                value: value.to_string(),
            })
        })
        .collect()
}

fn leakage_report(left: PathBuf, right: PathBuf) -> Result<Value, String> {
    let left_rows = read_manifest(&left)?;
    let right_rows = read_manifest(&right)?;
    let mut overlaps = BTreeMap::new();
    let mut matches = Vec::new();
    for field in [
        "id",
        "page_id",
        "image",
        "layout_id",
        "language_mix",
        "quality_profile",
    ] {
        let left_keys = key_set(&left_rows, field);
        let right_keys = key_set(&right_rows, field);
        overlaps.insert(
            field.to_string(),
            left_keys.intersection(&right_keys).count(),
        );
        use std::collections::HashMap;
        let mut right_map: HashMap<String, Vec<String>> = HashMap::new();
        for row in indexed_rows(&right_rows, field) {
            right_map.entry(row.value).or_default().push(row.id);
        }
        for left_row in indexed_rows(&left_rows, field) {
            if let Some(right_ids) = right_map.get(&left_row.value) {
                for right_id in right_ids {
                    matches.push(serde_json::json!({
                        "field": field,
                        "value": left_row.value,
                        "left_id": left_row.id,
                        "right_id": right_id,
                    }));
                }
            }
        }
    }
    let has_leakage = !matches.is_empty();
    Ok(serde_json::json!({
        "left": left.to_string_lossy(),
        "right": right.to_string_lossy(),
        "has_leakage": has_leakage,
        "overlaps": overlaps,
        "matches": matches,
    }))
}

pub fn leakage_check(left: PathBuf, right: PathBuf, out: Option<PathBuf>) -> Result<(), String> {
    let report = leakage_report(left, right)?;
    if let Some(out) = out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("cannot create leakage report dir: {err}"))?;
        }
        fs::write(
            &out,
            serde_json::to_string_pretty(&report).map_err(|err| err.to_string())?,
        )
        .map_err(|err| format!("cannot write leakage report: {err}"))?;
    }
    if !report["has_leakage"].as_bool().unwrap_or(false) {
        println!("no leakage detected");
        Ok(())
    } else {
        let overlaps = report
            .get("overlaps")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let errors: Vec<String> = overlaps
            .iter()
            .filter_map(|(field, count)| {
                let count = count.as_u64().unwrap_or(0);
                (count > 0).then(|| format!("{field} overlap: {count}"))
            })
            .collect();
        Err(errors.join("\n"))
    }
}
