use lambda_http::{
    run_concurrent, service_fn, tracing, Body, Error, Request, RequestExt, Response,
};
use sha2::{Digest, Sha256};

const MAX_DELAY_MS: u64 = 15_000;
const MAX_HASH_LOOPS: u64 = 1_000_000;

fn json_response(status: u16, body: serde_json::Value) -> Result<Response<Body>, Error> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::Text(body.to_string()))
        .map_err(|e| Box::new(e) as Error)
}

fn bad_request(message: impl Into<String>) -> Result<Response<Body>, Error> {
    json_response(400, serde_json::json!({ "error": message.into() }))
}

/// Handle API Gateway requests.
async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    let params = event.query_string_parameters();

    let delay_ms: u64 = match params.first("delay_ms") {
        None => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(v) if v <= MAX_DELAY_MS => v,
            Ok(_) => return bad_request(format!("delay_ms must be between 0 and {MAX_DELAY_MS}")),
            Err(_) => return bad_request("delay_ms must be an integer"),
        },
    };

    let hash_loops: u64 = match params.first("hash_loops") {
        None => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(v) if v <= MAX_HASH_LOOPS => v,
            Ok(_) => {
                return bad_request(format!("hash_loops must be between 0 and {MAX_HASH_LOOPS}"));
            }
            Err(_) => return bad_request("hash_loops must be an integer"),
        },
    };

    // Simulate I/O or downstream waiting.
    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

    // Exercise CPU by chaining SHA-256 hashes. Each loop hashes the previous digest,
    // so the optimizer can't remove the work and the loop can't be parallelized.
    let mut prev_digest: [u8; 32] = [0u8; 32];
    for i in 0..hash_loops {
        let mut hasher = Sha256::new();
        hasher.update(prev_digest);
        hasher.update(i.to_le_bytes());
        prev_digest = hasher.finalize().into();
    }

    let digest_hex: String = prev_digest.iter().map(|b| format!("{:02x}", b)).collect();

    json_response(
        200,
        serde_json::json!({
            "message": "Hello World!",
            "delay_ms": delay_ms,
            "hash_loops": hash_loops,
            "hash": digest_hex,
        }),
    )
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing::init_default_subscriber();

    run_concurrent(service_fn(function_handler)).await
}
