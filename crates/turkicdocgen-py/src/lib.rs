#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::PathBuf;

fn map_err(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

#[pyfunction]
fn dataset_summary(manifest: PathBuf, out: PathBuf) -> PyResult<()> {
    turkicdocgen_core::stats::dataset_summary(manifest, out).map_err(map_err)
}

#[pyfunction]
fn dedup_text(manifest: PathBuf, out: PathBuf) -> PyResult<()> {
    turkicdocgen_core::dedup::dedup_text(manifest, out).map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (manifest, out, threshold=None))]
fn dedup_text_minhash(manifest: PathBuf, out: PathBuf, threshold: Option<f64>) -> PyResult<()> {
    turkicdocgen_core::dedup::dedup_text_minhash(manifest, out, threshold).map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (manifest, images_root=None))]
fn validate_manifest(manifest: PathBuf, images_root: Option<PathBuf>) -> PyResult<()> {
    turkicdocgen_core::manifest::manifest_check(manifest, images_root).map_err(map_err)
}

#[pymodule]
#[pyo3(name = "_rustcore")]
fn turkicdocgen_rustcore(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dataset_summary, module)?)?;
    module.add_function(wrap_pyfunction!(dedup_text, module)?)?;
    module.add_function(wrap_pyfunction!(dedup_text_minhash, module)?)?;
    module.add_function(wrap_pyfunction!(validate_manifest, module)?)?;
    Ok(())
}
