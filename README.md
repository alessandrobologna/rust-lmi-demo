# Rust Lambda Managed Instances (LMI) Demo

A demonstration comparing AWS Lambda with standard execution vs Lambda Managed Instances (LMI) capacity provider, using a Rust-based serverless application.

## Overview

This project deploys three Rust Lambda functions behind API Gateway:

| Endpoint | Memory | Capacity Provider | Description |
|----------|--------|-------------------|-------------|
| `/default/128/hello` | 128 MB | Standard Lambda | Baseline for I/O(wait) workload |
| `/default/2048/hello` | 2048 MB | Standard Lambda | Baseline for CPU workload |
| `/lmi/2048/hello` | 2048 MB | LMI | Lambda Managed Instances with concurrent execution |

The LMI function uses a [forked aws-lambda-rust-runtime](https://github.com/alessandrobologna/aws-lambda-rust-runtime/tree/feat/concurrent-lambda-runtime) that supports handling multiple concurrent requests within a single execution environment.

## Architecture

```mermaid
flowchart LR
    Client["Client<br/>(k6)"] --> API["API<br/>Gateway"]
    API --> Lambda128["Lambda (128)<br/>128 MB, Standard"]
    API --> Lambda2048["Lambda (2048)<br/>2048 MB, Standard"]
    API --> LMI["Lambda (LMI)<br/>2048 MB, 64 concurrent<br/>per execution env"]
```

### LMI Configuration

The LMI Lambda is configured with:
- `ExecutionEnvironmentMemoryGiBPerVCpu: 2` - Memory per vCPU ratio
- `PerExecutionEnvironmentMaxConcurrency: 64` - Up to 64 concurrent requests per execution environment
- Capacity provider instance type: `c8g.xlarge` (arm64)
- Capacity provider subnets: 2 (the first two IDs from `ManagedInstancesSubnetIds`)
- Capacity provider baseline during tests: 2 instances (scale-in target)

## Prerequisites

- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- [Rust](https://rustup.rs/) with [cargo-lambda](https://github.com/cargo-lambda/cargo-lambda)
- [k6](https://k6.io/docs/get-started/installation/) for load testing
- [uv](https://docs.astral.sh/uv/) for running the benchmark script
- AWS account with VPC subnets and security group for LMI

## Deployment

```bash
sam build
sam deploy --guided
```

You'll be prompted for:
- `ManagedInstancesSubnetIds` - VPC subnet IDs for LMI capacity provider
- `ManagedInstancesSecurityGroupId` - Security group ID for LMI

## Load Testing

Run the benchmark to execute all scenarios. Each scenario compares all three endpoints (`/default/128/hello`, `/default/2048/hello`, `/lmi/2048/hello`).

```bash
uv run benchmark.py --stack rust-lmi-demo --scenario all
```

By default, when running `--scenario all`, the script waits between scenarios until the LMI capacity provider scales back to its pre-run EC2 instance count (so each scenario starts from a comparable baseline). Disable with `--no-wait-for-scale-in`.

### Scenarios

| Scenario | `--scenario` | Workload |
|----------|--------------|----------|
| Fast I/O | `fast-io` | `delay_ms=100` |
| Slow I/O | `slow-io` | `delay_ms=500` |
| Low CPU | `low-cpu` | `hash_loops=50000` |
| High CPU | `high-cpu` | `hash_loops=200000` |

### Phases (k6 stages)

Default stages are:
- Ramp-up: `0→64` VUs (3 minutes)
- Steady: `64→64` VUs (6 minutes)
- Ramp-up: `64→128` VUs (3 minutes) — goal is to push until something breaks

The benchmark uses per-endpoint k6 scenarios so a slow endpoint (often `/default/128/hello` on CPU) does not throttle load against the others.
Stage targets apply per endpoint; peak total k6 VUs will be ~3× the target because there are 3 endpoints.

You can override stages with `--stage-targets` (single duration repeated for each target) or `--stages-json` (full control).

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--stack` | (required) | CloudFormation stack name |
| `--region` | AWS default | AWS region |
| `--duration` | 30s | Duration per k6 stage (only used with `--stage-targets`) |
| `--stage-targets` | (none) | Comma-separated k6 stage targets, e.g. `20,20,40` |
| `--stages-json` | (none) | Full k6 stages JSON (advanced) |
| `--scenario` | all | `fast-io`, `slow-io`, `low-cpu`, `high-cpu`, or `all` |
| `--cpu-low-hash-loops` | 50000 | `hash_loops` used for the Low CPU scenario (max 1,000,000) |
| `--cpu-high-hash-loops` | 200000 | `hash_loops` used for the High CPU scenario (max 1,000,000) |
| `--io-fast-delay-ms` | 100 | `delay_ms` used for the Fast I/O scenario (max 15,000) |
| `--io-slow-delay-ms` | 500 | `delay_ms` used for the Slow I/O scenario (max 15,000) |
| `--output-dir` | `./benchmark-results` | Directory for results |
| `--skip-test` | false | Regenerate charts from existing CSV |
| `--wait-for-scale-in/--no-wait-for-scale-in` | (auto) | Wait between scenarios for LMI scale-in (defaults to on for `--scenario all`) |
| `--capacity-provider-arn` | (derived) | Capacity provider ARN used to find EC2 instances for the wait loop |
| `--scale-in-check-seconds` | 60 | How often to check instance count while waiting |
| `--scale-in-max-minutes` | 90 | Max time to wait for scale-in between scenarios |

## Results

The benchmark writes one CSV + one chart per scenario to `benchmark-results/`.

### Latest Benchmarks (2025-12-19, per-endpoint mode, stages: 0→64 (3m), 64→64 (6m), 64→128 (3m))

#### Fast I/O (`delay_ms=100`)

```
Endpoint         Reqs    5xx   Avg(ms)     p95     p99     Max
--------------------------------------------------------------
128            180220      0     154.8   169.1   195.0   839.6
2048           181808      0     152.6   164.1   190.3   889.6
LMI            182603      0     151.5   163.4   187.8  1711.0
```

<img src="benchmark-results/benchmark-fast-io-20251219-020111.png" alt="Fast I/O Benchmark Results" width="100%">

#### Slow I/O (`delay_ms=500`)

```
Endpoint         Reqs    5xx   Avg(ms)     p95     p99      Max
---------------------------------------------------------------
128             68438      0     571.8   629.1   938.8    993.6
2048            68485      0     571.3   629.7   938.0   1450.3
LMI             68494      0     571.2   630.6   937.5    995.7
```

<img src="benchmark-results/benchmark-slow-io-20251219-020111.png" alt="Slow I/O Benchmark Results" width="100%">

#### Low CPU (`hash_loops=50000`)

```
Endpoint         Reqs    5xx   Avg(ms)     p95     p99     Max
--------------------------------------------------------------
128            133994      0     242.5   364.3   427.0   948.3
2048           250635      0      82.8   153.2   262.2   968.7
LMI            218900      0     109.3   205.1   307.0   880.8
```

<img src="benchmark-results/benchmark-low-cpu-20251219-020111.png" alt="Low CPU Benchmark Results" width="100%">

#### High CPU (`hash_loops=200000`)

```
Endpoint         Reqs    5xx   Avg(ms)     p95     p99      Max
---------------------------------------------------------------
128             52342      0     778.8   902.1   951.5   1341.2
2048           202870      0     125.9   200.2   321.3   1205.5
LMI            172266    209     166.1   352.2   497.7    900.9
```

<img src="benchmark-results/benchmark-high-cpu-20251219-031820.png" alt="High CPU Benchmark Results" width="100%">

### Notes

- CPU-heavy work does not benefit from high per-environment concurrency. With `ExecutionEnvironmentMemoryGiBPerVCpu: 2` and a 2 GB function, each execution environment gets ~1 vCPU; allowing up to `PerExecutionEnvironmentMaxConcurrency: 64` means CPU-bound requests contend for that vCPU.
- Lambda managed instances capacity providers scale gradually; AWS docs say they maintain enough headroom for traffic to double within 5 minutes. If traffic increases faster than this, requests can be throttled. ([AWS docs](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-scaling.html))
- If you don’t wait for LMI scale-in between scenarios, later scenarios will start from a pre-scaled capacity provider and won’t show scale-up/backlog behavior from idle. The benchmark can wait automatically when running `--scenario all` (disable with `--no-wait-for-scale-in`).
- The latency scatter plot is downsampled and the points are shuffled (per endpoint) to reduce overdraw; the CSV contains the full dataset.

## Project Structure

```
rust-lmi-demo/
├── src/main.rs           # Lambda handler
├── Cargo.toml            # Dependencies (uses forked runtime)
├── Cargo.lock            # Cargo lockfile
├── template.yaml         # SAM template
├── loadtest.js           # k6 load test script
├── benchmark.py          # Python orchestration + charting
└── benchmark-results/    # CSV data and PNG charts
```

## How It Works

### Standard Lambda
Each request gets its own Lambda invocation. Under load, AWS scales by creating more execution environments.

### LMI with Concurrent Runtime
The LMI capacity provider, combined with the concurrent Rust runtime, allows a single execution environment to handle multiple requests simultaneously using Tokio's async runtime. This reduces:
- Cold start frequency (fewer environments needed)
- Per-request overhead
- Total Lambda invocations under load

## Cleanup

```bash
sam delete --stack-name rust-lmi-demo
```

## License

MIT
