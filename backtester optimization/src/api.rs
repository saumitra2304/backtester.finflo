use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde_json::json;
use std::sync::Mutex;

// Assuming backtester.rs is in the same project and the necessary structs and functions are public
mod backtester;
use backtester::{backtesting, Config};

async fn backtest_handler(item: web::Json<Config>) -> impl Responder {
    let config_str = serde_json::to_string(&*item).unwrap();
    match backtesting(&config_str) {
        Ok((equity_curve, trades)) => {
            HttpResponse::Ok().json(json!({
                "equity_curve": equity_curve,
                "trades": trades
            }))
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(web::resource("/backtest").route(web::post().to(backtest_handler)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

