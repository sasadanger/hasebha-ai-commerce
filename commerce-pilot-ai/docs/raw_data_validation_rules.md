# Raw Data Validation Rules

## Universal rules

Validation operates on one explicitly selected dataset directory and must not read another domain. It records observed facts and does not repair, normalize, or reinterpret raw data.

For the selected dataset:

- The configured raw directory must exist and contain at least one non-partial file.
- Every inspected file must be readable and non-empty.
- ZIP archives must open successfully and pass member integrity checks.
- Compressed JSON Lines must decompress and its first non-empty record must parse as JSON.
- Extraction must reject paths that would escape the selected extraction directory.
- Validation failures must identify the affected file and cause.
- A successful validation does not certify analytical fitness, licensing, completeness, or semantic correctness.

## Immutable-input checks

Locally calculated SHA-256 checksums belong in generated profiles and provenance records. Repeated runs should compare checksums before accepting an existing raw file. Source changes require a new provenance entry; silent replacement is prohibited.

## Dataset-specific profiling goals

### Olist

Observe archive integrity, file names, sizes, checksums, CSV row counts, and CSV headers. Header names are reported as observed source facts, not predeclared schema.

### Instacart

Observe archive integrity, file names, sizes, checksums, CSV row counts, and CSV headers after authorized acquisition. Do not infer customer identity or connect records outside this dataset.

### Amazon Reviews 2023 — Appliances

Observe gzip readability, JSON Lines record count, keys observed in records, file size, and checksum. Do not reproduce review content in profile reports and do not inspect other Amazon categories.

## Observed facts versus assumptions

An observed fact is directly measured from acquired bytes, such as file size, checksum, parse success, record count, or header text. A source fact is explicitly stated by a verified publisher page. Anything else is an assumption and must be labeled as such or omitted. Validation must not turn plausible expectations into claimed facts.

## Gate

No feature engineering, dataset transformation, model development, or business inference may begin until the selected dataset has documented provenance and passes the applicable raw validation.

