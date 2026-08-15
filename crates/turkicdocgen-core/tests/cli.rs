use std::fs;
use std::process::Command;
use tempfile::tempdir;

use serde_json::json;

fn bin() -> String {
    env!("CARGO_BIN_EXE_turkicdocgen-core").to_string()
}

fn row(id: &str, image: &str, text: &str) -> String {
    let val = json!({
        "id": id,
        "page_id": id,
        "image": image,
        "image_path": image,
        "layout_id": "official_letter_page",
        "language_mix": "kk",
        "quality_profile": "scan_light",
        "effect_metadata": {
            "effects": ["scan_light"]
        },
        "qa_ok": true,
        "qa_issues": [],
        "zones": [{
            "zone_id": "body",
            "zone_type": "body",
            "bbox": [10, 10, 200, 80],
            "text": text,
            "language": "kk",
            "reading_order": 1
        }]
    });
    serde_json::to_string(&val).unwrap()
}

fn write_manifest(lines: &[String]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let manifest = dir.path().join("manifest.jsonl");
    fs::write(&manifest, lines.join("\n")).unwrap();
    (dir, manifest)
}

#[test]
fn manifest_check_accepts_ocr_core_manifest() {
    let (_dir, manifest) = write_manifest(&[row("a", "images/a.jpg", "Kazakh sample ә қ")]);
    let output = Command::new(bin())
        .args(["manifest-check", "--manifest", manifest.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("manifest ok: 1 rows"));
}

#[test]
fn schema_manifest_check_rejects_missing_zones() {
    let dir = tempdir().unwrap();
    let manifest = dir.path().join("manifest.jsonl");
    fs::write(
        &manifest,
        r#"{"id":"a","page_id":"a","image":"images/a.jpg","image_path":"images/a.jpg","layout_id":"official_letter_page","language_mix":"kk","quality_profile":"clean","effect_metadata":{},"qa_ok":true,"qa_issues":[],"zones":[]}"#,
    )
    .unwrap();
    let output = Command::new(bin())
        .args([
            "schema-manifest-check",
            "--manifest",
            manifest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("zones_empty_or_invalid"));
}

#[test]
fn dataset_summary_reports_layout_language_and_quality() {
    let (dir, manifest) = write_manifest(&[
        row("a", "images/a.jpg", "Kazakh sample ә қ"),
        row("b", "images/b.jpg", "Kyrgyz sample ө ү"),
    ]);
    let out = dir.path().join("summary.json");
    let output = Command::new(bin())
        .args([
            "dataset-summary",
            "--manifest",
            manifest.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = fs::read_to_string(out).unwrap();
    assert!(text.contains("by_layout"));
    assert!(text.contains("official_letter_page"));
    assert!(text.contains("scan_light"));
}

#[test]
fn dedup_text_uses_zone_text() {
    let (dir, manifest) = write_manifest(&[
        row("a", "images/a.jpg", "Duplicate body text"),
        row("b", "images/b.jpg", "Duplicate body text"),
        row("c", "images/c.jpg", "Different body text"),
    ]);
    let out = dir.path().join("dedup.json");
    let output = Command::new(bin())
        .args([
            "dedup-text",
            "--manifest",
            manifest.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = fs::read_to_string(out).unwrap();
    assert!(text.contains("\"duplicate_text_hashes\": 1"));
}

#[test]
fn manifest_check_rejects_missing_image_when_root_is_provided() {
    let (dir, manifest) = write_manifest(&[row("a", "images/missing.jpg", "Body text")]);
    let images_root = dir.path();
    let output = Command::new(bin())
        .args([
            "manifest-check",
            "--manifest",
            manifest.to_str().unwrap(),
            "--images-root",
            images_root.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("image does not exist"));
}
