use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use ttf_parser::Face;

pub fn glyph_check(font: PathBuf, text: String) -> Result<(), String> {
    let data = fs::read(&font).map_err(|err| format!("cannot read font: {err}"))?;
    let face = Face::parse(&data, 0).map_err(|err| format!("cannot parse font: {err:?}"))?;
    let mut seen = HashSet::new();
    let mut missing = Vec::new();

    for ch in text.chars() {
        if ch.is_whitespace() || !seen.insert(ch) {
            continue;
        }
        if face.glyph_index(ch).is_none() {
            missing.push(ch.to_string());
        }
    }

    let payload = json!({
        "ok": missing.is_empty(),
        "font_path": font.to_string_lossy(),
        "missing_chars": missing,
    });
    println!(
        "{}",
        serde_json::to_string(&payload)
            .map_err(|err| format!("cannot serialize glyph report: {err}"))?
    );

    if missing.is_empty() {
        Ok(())
    } else {
        Err("glyph coverage failed".to_string())
    }
}
