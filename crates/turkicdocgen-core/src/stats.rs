use crate::util::{duplicate_count, increment_count, read_manifest, string_field, zone_text};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const SPECIAL_CHARS: &[&str] = &[
    "\u{04D8}", "\u{04D9}", "\u{0492}", "\u{0493}", "\u{049A}", "\u{049B}", "\u{04A0}", "\u{04A1}",
    "\u{04A2}", "\u{04A3}", "\u{04E8}", "\u{04E9}", "\u{04B0}", "\u{04B1}", "\u{04AE}", "\u{04AF}",
    "\u{0406}", "\u{0456}", "\u{0496}", "\u{0497}", "\u{0498}", "\u{0499}", "\u{04AA}", "\u{04AB}",
    "\u{04BA}", "\u{04BB}",
];

fn special_char_counts(rows: &[Value]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        let text = zone_text(row);
        if !text.is_empty() {
            for pattern in SPECIAL_CHARS {
                let count = text.matches(pattern).count();
                if count > 0 {
                    *counts.entry((*pattern).to_string()).or_insert(0) += count;
                }
            }
        }
    }
    counts
}

pub fn special_char_stats(path: PathBuf) -> Result<(), String> {
    let rows = read_manifest(&path)?;
    for (ch, count) in special_char_counts(&rows) {
        println!("{ch}\t{count}");
    }
    Ok(())
}

pub fn dataset_summary(manifest: PathBuf, out: PathBuf) -> Result<(), String> {
    let rows = read_manifest(&manifest)?;
    let mut by_layout = BTreeMap::new();
    let mut by_quality = BTreeMap::new();
    let mut by_language = BTreeMap::new();

    for row in &rows {
        increment_count(
            &mut by_layout,
            string_field(row, "layout_id").unwrap_or("unknown"),
        );
        increment_count(
            &mut by_quality,
            string_field(row, "quality_profile").unwrap_or("unknown"),
        );
        increment_count(
            &mut by_language,
            string_field(row, "language_mix").unwrap_or("unknown"),
        );
    }

    let special_counts = special_char_counts(&rows);
    let payload = serde_json::json!({
        "manifest": manifest.to_string_lossy(),
        "total_rows": rows.len(),
        "by_layout": by_layout,
        "by_quality": by_quality,
        "by_language": by_language,
        "special_char_counts": special_counts,
        "duplicate_images": duplicate_count(&rows, "image"),
    });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("cannot create summary dir: {err}"))?;
    }
    fs::write(
        &out,
        serde_json::to_string_pretty(&payload).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("cannot write dataset summary: {err}"))?;
    println!("dataset summary written: {}", out.to_string_lossy());
    Ok(())
}
