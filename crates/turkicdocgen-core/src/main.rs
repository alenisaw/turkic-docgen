use std::env;
use turkicdocgen_core::{
    dedup, glyph, hashing, leakage, manifest, perceptual, schema_manifest, split, stats, util,
};

fn usage() -> String {
    [
        "usage:",
        "  turkicdocgen-core manifest-check --manifest <path>",
        "  turkicdocgen-core validate-manifest --manifest <path> [--images-root <path>]",
        "  turkicdocgen-core schema-manifest-check --manifest <path> [--images-root <path>]",
        "  turkicdocgen-core glyph-check --font <path> --text <required-chars>",
        "  turkicdocgen-core image-ahash --image <path>",
        "  turkicdocgen-core special-char-stats --manifest <path>",
        "  turkicdocgen-core special-stats --manifest <path>",
        "  turkicdocgen-core leakage-check --left <path> --right <path>",
        "  turkicdocgen-core leakage-check --train <path> --bench <path> --out <path>",
        "  turkicdocgen-core hash-files --manifest <path> --out hashes.jsonl",
        "  turkicdocgen-core dedup-text --manifest <path> --out dedup_report.json",
        "  turkicdocgen-core dedup-text-minhash --manifest <path> --out dedup_minhash_report.json [--threshold <0..1>]",
        "  turkicdocgen-core dataset-summary --manifest <path> --out <path>",
        "  turkicdocgen-core split-manifest --manifest <path> --train-out <path> --validation-out <path>",
    ]
    .join("\n")
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.len() < 2 {
        return Err(usage());
    }
    match args[1].as_str() {
        "manifest-check" | "validate-manifest" => {
            let images_root = util::value_after(&args, "--images-root").ok();
            manifest::manifest_check(util::value_after(&args, "--manifest")?, images_root)
        }
        "special-char-stats" | "special-stats" => {
            stats::special_char_stats(util::value_after(&args, "--manifest")?)
        }
        "schema-manifest-check" => schema_manifest::schema_manifest_check(
            util::value_after(&args, "--manifest")?,
            util::value_after(&args, "--images-root").ok(),
        ),
        "glyph-check" => glyph::glyph_check(
            util::value_after(&args, "--font")?,
            util::string_after(&args, "--text")?,
        ),
        "image-ahash" => perceptual::image_ahash(util::value_after(&args, "--image")?),
        "leakage-check" => {
            let left = util::value_after(&args, "--left")
                .or_else(|_| util::value_after(&args, "--train"))?;
            let right = util::value_after(&args, "--right")
                .or_else(|_| util::value_after(&args, "--bench"))?;
            let out = args
                .iter()
                .position(|arg| arg == "--out")
                .and_then(|pos| args.get(pos + 1))
                .map(std::path::PathBuf::from);
            leakage::leakage_check(left, right, out)
        }
        "hash-files" => hashing::hash_files(
            util::value_after(&args, "--manifest")?,
            util::value_after(&args, "--out")?,
        ),
        "dedup-text" => dedup::dedup_text(
            util::value_after(&args, "--manifest")?,
            util::value_after(&args, "--out")?,
        ),
        "dedup-text-minhash" => {
            let threshold = util::optional_string_after(&args, "--threshold")
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|err| format!("invalid --threshold: {err}"))
                })
                .transpose()?;
            dedup::dedup_text_minhash(
                util::value_after(&args, "--manifest")?,
                util::value_after(&args, "--out")?,
                threshold,
            )
        }
        "dataset-summary" => stats::dataset_summary(
            util::value_after(&args, "--manifest")?,
            util::value_after(&args, "--out")?,
        ),
        "split-manifest" => {
            let split_ratio = util::optional_string_after(&args, "--split-ratio")
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|err| format!("invalid --split-ratio: {err}"))
                })
                .transpose()?;
            split::split_manifest(
                util::value_after(&args, "--manifest")?,
                util::value_after(&args, "--train-out")?,
                util::value_after(&args, "--validation-out")?,
                split_ratio,
            )
        }
        _ => Err(usage()),
    }
}

fn main() {
    if let Err(err) = run(env::args().collect()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
