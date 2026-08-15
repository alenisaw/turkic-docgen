use crate::util::read_manifest;
use std::fs;
use std::path::PathBuf;

fn deterministic_shuffle<T>(vec: &mut [T]) {
    let mut seed: u64 = 42;
    let n = vec.len();
    if n <= 1 {
        return;
    }
    for i in (1..n).rev() {
        seed = (seed.wrapping_mul(6364136223846793005).wrapping_add(1)) ^ seed;
        let j = (seed % (i as u64 + 1)) as usize;
        vec.swap(i, j);
    }
}

pub fn split_manifest(
    manifest: PathBuf,
    train_out: PathBuf,
    validation_out: PathBuf,
    split_ratio: Option<f64>,
) -> Result<(), String> {
    let mut rows = read_manifest(&manifest)?;
    deterministic_shuffle(&mut rows);
    let ratio = split_ratio.unwrap_or(0.85);
    if !(0.0..=1.0).contains(&ratio) {
        return Err("split ratio must be between 0.0 and 1.0".to_string());
    }
    let split_at = ((rows.len() as f64) * ratio).round() as usize;
    let mut train = String::new();
    let mut validation = String::new();
    for (idx, row) in rows.iter().enumerate() {
        let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
        if idx < split_at {
            train.push_str(&line);
            train.push('\n');
        } else {
            validation.push_str(&line);
            validation.push('\n');
        }
    }
    fs::write(train_out, train).map_err(|err| format!("cannot write train split: {err}"))?;
    fs::write(validation_out, validation)
        .map_err(|err| format!("cannot write validation split: {err}"))?;
    println!(
        "split ok: {split_at} train, {} validation",
        rows.len() - split_at
    );
    Ok(())
}
