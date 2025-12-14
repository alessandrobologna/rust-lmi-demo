use lambda_http::{Body, Error, Request, RequestExt, Response, run_concurrent, service_fn, tracing};

/// Handle API Gateway requests.
async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    // Get delay_ms from query parameter, default to 0 if not provided
    let delay_ms: u64 = event
        .query_string_parameters()
        .first("delay_ms")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    let resp = Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(Body::from(r#"{"message": "Hello World!"}"#))
        .map_err(Box::new)?;

    Ok(resp)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing::init_default_subscriber();

    run_concurrent(service_fn(function_handler)).await
}
