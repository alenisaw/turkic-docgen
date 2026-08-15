use crate::util::{read_manifest, string_field};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

pub fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|err| format!("cannot open file for hash: {err}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 64];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("cannot read file for hash: {err}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_manifest_file(manifest: &std::path::Path, value: &str) -> Result<PathBuf, String> {
    let parent = manifest
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    crate::util::resolve_path_safe(parent, value)
}

pub fn hash_files(manifest: PathBuf, out: PathBuf) -> Result<(), String> {
    let rows = read_manifest(&manifest)?;
    let mut output = String::new();
    for row in rows {
        let sample_id = string_field(&row, "id").unwrap_or("");
        for field in ["image", "image_path"] {
            let Some(value) = string_field(&row, field) else {
                continue;
            };
            let path = resolve_manifest_file(&manifest, value)?;
            let (status, sha256) = if path.exists() {
                ("ok", sha256_file(&path)?)
            } else {
                ("missing", String::new())
            };
            let record = serde_json::json!({
                "id": sample_id,
                "field": field,
                "path": value,
                "status": status,
                "sha256": sha256,
            });
            output.push_str(&serde_json::to_string(&record).map_err(|err| err.to_string())?);
            output.push('\n');
        }
    }
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create hash report dir: {err}"))?;
    }
    fs::write(&out, output).map_err(|err| format!("cannot write hash report: {err}"))?;
    println!("hash report written: {}", out.to_string_lossy());
    Ok(())
}
