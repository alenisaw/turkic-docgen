use crate::util::{increment_count, read_manifest, string_field, zone_text};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

const MINHASH_SIZE: usize = 64;
const DEFAULT_THRESHOLD: f64 = 0.82;
const SHINGLE_SIZE: usize = 5;

pub fn dedup_text(manifest: PathBuf, out: PathBuf) -> Result<(), String> {
    let rows = read_manifest(&manifest)?;
    let mut text_hash_counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        let text = zone_text(row);
        if !text.trim().is_empty() {
            increment_count(&mut text_hash_counts, sha256_text(&text));
        }
    }
    let duplicate_groups: Vec<Value> = text_hash_counts
        .iter()
        .filter(|(_hash, count)| **count > 1)
        .map(|(hash, count)| serde_json::json!({"text_hash": hash, "count": count}))
        .collect();
    let payload = serde_json::json!({
        "manifest": manifest.to_string_lossy(),
        "total_rows": rows.len(),
        "duplicate_text_hashes": duplicate_groups.iter().map(|item| item["count"].as_u64().unwrap_or(0) - 1).sum::<u64>(),
        "duplicate_groups": duplicate_groups,
    });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create dedup report dir: {err}"))?;
    }
    fs::write(
        &out,
        serde_json::to_string_pretty(&payload).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("cannot write dedup report: {err}"))?;
    println!("dedup report written: {}", out.to_string_lossy());
    Ok(())
}

#[derive(Debug)]
struct TextSignature {
    row_index: usize,
    id: String,
    text_hash: String,
    token_count: usize,
    signature: [u64; MINHASH_SIZE],
}

pub fn dedup_text_minhash(
    manifest: PathBuf,
    out: PathBuf,
    threshold: Option<f64>,
) -> Result<(), String> {
    let threshold = threshold.unwrap_or(DEFAULT_THRESHOLD);
    if !(0.0..=1.0).contains(&threshold) {
        return Err("--threshold must be between 0 and 1".to_string());
    }
    let rows = read_manifest(&manifest)?;
    let mut signatures = Vec::new();
    let mut skipped_rows = 0usize;
    for (row_index, row) in rows.iter().enumerate() {
        let text = zone_text(row);
        if text.trim().is_empty() {
            skipped_rows += 1;
            continue;
        }
        let tokens = normalized_tokens(&text);
        if tokens.is_empty() {
            skipped_rows += 1;
            continue;
        }
        signatures.push(TextSignature {
            row_index,
            id: string_field(row, "id")
                .or_else(|| string_field(row, "sample_id"))
                .unwrap_or("")
                .to_string(),
            text_hash: sha256_text(&text),
            token_count: tokens.len(),
            signature: minhash_signature(&tokens),
        });
    }

    // Locality Sensitive Hashing (LSH) Bucketing
    // 16 bands of 4 rows = 64 MinHash values
    let num_bands = 16;
    let rows_per_band = 4;
    let mut candidate_pairs = HashSet::new();

    for band in 0..num_bands {
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (sig_idx, sig) in signatures.iter().enumerate() {
            let start = band * rows_per_band;
            let slice = &sig.signature[start..start + rows_per_band];

            // Generate a simple stable hash for the band rows
            let mut hasher = Sha256::new();
            for &val in slice {
                hasher.update(val.to_le_bytes());
            }
            let hash_bytes = hasher.finalize();
            let mut bucket_hash = 0u64;
            for i in 0..8 {
                bucket_hash = (bucket_hash << 8) | u64::from(hash_bytes[i]);
            }
            buckets.entry(bucket_hash).or_default().push(sig_idx);
        }

        for indices in buckets.values() {
            if indices.len() > 1 {
                for i in 0..indices.len() {
                    for j in (i + 1)..indices.len() {
                        let left = indices[i];
                        let right = indices[j];
                        if left < right {
                            candidate_pairs.insert((left, right));
                        } else {
                            candidate_pairs.insert((right, left));
                        }
                    }
                }
            }
        }
    }

    // Shuffle candidates deterministically to prevent row index bias when capping
    let mut candidate_list: Vec<(usize, usize)> = candidate_pairs.into_iter().collect();
    deterministic_shuffle(&mut candidate_list);

    let mut pairs = Vec::new();
    let max_pairs = 100_000;
    let limit_reached = candidate_list.len() > max_pairs;
    let comparison_limit = candidate_list.len().min(max_pairs);

    if limit_reached {
        eprintln!("Warning: reached max pairs cap ({max_pairs}), stopping comparison early to prevent DoS");
    }

    for &(left_index, right_index) in candidate_list.iter().take(comparison_limit) {
        let left = &signatures[left_index];
        let right = &signatures[right_index];
        let score = minhash_similarity(&left.signature, &right.signature);
        if score >= threshold {
            pairs.push(serde_json::json!({
                "left_row": left.row_index,
                "right_row": right.row_index,
                "left_id": left.id,
                "right_id": right.id,
                "left_text_hash": left.text_hash,
                "right_text_hash": right.text_hash,
                "left_token_count": left.token_count,
                "right_token_count": right.token_count,
                "score": round_score(score),
            }));
        }
    }

    // Sort final pairs for stable report ordering
    pairs.sort_by_key(|p| {
        (
            p["left_row"].as_u64().unwrap_or(0),
            p["right_row"].as_u64().unwrap_or(0),
        )
    });

    let payload = serde_json::json!({
        "manifest": manifest.to_string_lossy(),
        "algorithm": "token_shingle_minhash_v1",
        "threshold": threshold,
        "signature_size": MINHASH_SIZE,
        "shingle_size": SHINGLE_SIZE,
        "total_rows": rows.len(),
        "indexed_rows": signatures.len(),
        "skipped_rows": skipped_rows,
        "near_duplicate_pairs": pairs.len(),
        "pairs_limit_reached": limit_reached,
        "pairs": pairs,
    });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create minhash dedup report dir: {err}"))?;
    }
    fs::write(
        &out,
        serde_json::to_string_pretty(&payload).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("cannot write minhash dedup report: {err}"))?;
    println!("minhash dedup report written: {}", out.to_string_lossy());
    Ok(())
}

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

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn normalized_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn minhash_signature(tokens: &[String]) -> [u64; MINHASH_SIZE] {
    let mut signature = [u64::MAX; MINHASH_SIZE];
    let shingle_count = tokens.len().saturating_sub(SHINGLE_SIZE).saturating_add(1);
    for start in 0..shingle_count {
        let end = (start + SHINGLE_SIZE).min(tokens.len());
        let shingle = tokens[start..end].join("\u{1f}");
        for (seed, slot) in signature.iter_mut().enumerate() {
            let hash = stable_hash(seed as u64, shingle.as_bytes());
            if hash < *slot {
                *slot = hash;
            }
        }
    }
    signature
}

fn stable_hash(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed.wrapping_mul(0x9e3779b97f4a7c15);
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= hash >> 32;
    }
    hash
}

fn minhash_similarity(left: &[u64; MINHASH_SIZE], right: &[u64; MINHASH_SIZE]) -> f64 {
    let matches = left
        .iter()
        .zip(right.iter())
        .filter(|(left_hash, right_hash)| left_hash == right_hash)
        .count();
    matches as f64 / MINHASH_SIZE as f64
}

fn round_score(score: f64) -> f64 {
    (score * 10000.0).round() / 10000.0
}
