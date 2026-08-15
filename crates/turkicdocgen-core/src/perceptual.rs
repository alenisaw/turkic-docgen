use image::imageops::FilterType;
use serde_json::json;
use std::path::PathBuf;

fn average_hash64(image: &image::DynamicImage) -> u64 {
    let tiny = image.resize_exact(8, 8, FilterType::Triangle).to_luma8();
    let pixels = tiny
        .pixels()
        .map(|pixel| u32::from(pixel[0]))
        .collect::<Vec<_>>();
    assert!(!pixels.is_empty(), "pixels cannot be empty");
    let mean = pixels.iter().sum::<u32>() / pixels.len() as u32;
    pixels.iter().enumerate().fold(0_u64, |bits, (idx, value)| {
        if *value >= mean {
            bits | (1_u64 << idx)
        } else {
            bits
        }
    })
}

pub fn image_ahash(image_path: PathBuf) -> Result<(), String> {
    let image = image::open(&image_path).map_err(|err| format!("cannot read image: {err}"))?;
    let hash = average_hash64(&image);
    let payload = json!({
        "image_path": image_path.to_string_lossy(),
        "algorithm": "average_hash_8x8_luma",
        "hash": format!("{hash:016x}"),
    });
    println!(
        "{}",
        serde_json::to_string(&payload)
            .map_err(|err| format!("cannot serialize perceptual hash report: {err}"))?
    );
    Ok(())
}
