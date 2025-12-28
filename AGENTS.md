# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Rust Lambda handler (entrypoint in `src/main.rs`).
- `template.yaml` defines the SAM application, API routes, and LMI capacity provider.
- `benchmark.py` orchestrates k6 runs, parses CSVs, and renders charts.
- `loadtest.js` is the k6 script (arrival‑rate scenarios).
- `benchmark-results/` stores PNG charts (CSV data is gitignored).
- `Cargo.toml` / `Cargo.lock` manage Rust dependencies.

## Build, Test, and Development Commands
- `cargo build` — compile the Rust Lambda locally.
- `cargo test` — run Rust tests (if present).
- `sam build` — build the SAM application (Rust + Lambda).
- `sam deploy --guided` — deploy the stack and parameters.
- `uv run benchmark.py --stack rust-lmi-demo --scenario all --region us-east-1` — run all benchmark scenarios.
- `uv run benchmark.py --skip-test --cloudwatch-concurrency --refresh-cloudwatch --region us-east-1` — re‑fetch CloudWatch data and re‑plot charts without rerunning k6.

## Coding Style & Naming Conventions
- Rust follows standard `rustfmt` conventions; keep functions and variables descriptive.
- Python uses 4‑space indentation and type hints where practical.
- k6 scenario names use kebab‑case (`bursty-io`, `cpu-break`); endpoints are `128`, `512`, and `LMI`.
- Prefer ASCII in files unless existing content requires Unicode.

## Testing Guidelines
- No dedicated test framework is configured for this repo.
- If you add tests, use Rust’s standard `#[test]` in `src/` or `tests/`.
- Benchmark validation is via k6 CSVs + charts; do not commit CSVs.

## Commit & Pull Request Guidelines
- Commit messages use a short imperative summary, then a blank line with bullet points describing changes.
  - Example:
    - `Switch benchmarks to arrival-rate scenarios`
    - `- Update template to 512MB standard Lambda`
    - `- Regenerate benchmark charts`
- PRs should include:
  - Brief summary of changes
  - Updated charts or notes if benchmark behavior changed
  - Any stack/config changes (e.g., capacity provider limits)

## Security & Configuration Tips
- AWS credentials are required for deploys and CloudWatch queries; avoid committing secrets.
- `benchmark-results/*.csv` is gitignored—use it locally for analysis only.
- Capacity provider limits are intentionally constrained to expose break‑points; adjust `template.yaml` if testing scale‑out behavior.
