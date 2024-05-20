use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use yata::methods::{EMA, SMA};
use yata::prelude::*;
use reqwest::blocking::Client;
use reqwest::Error;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[derive(Debug, Deserialize)]
pub struct MarketData {
    dates: Vec<NaiveDateTime>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
}

impl MarketData {
    fn new(capacity: usize) -> Self {
        Self {
            dates: Vec::with_capacity(capacity),
            opens: Vec::with_capacity(capacity),
            highs: Vec::with_capacity(capacity),
            lows: Vec::with_capacity(capacity),
            closes: Vec::with_capacity(capacity),
            volumes: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, date: NaiveDateTime, open: f64, high: f64, low: f64, close: f64, volume: f64) {
        self.dates.push(date);
        self.opens.push(open);
        self.highs.push(high);
        self.lows.push(low);
        self.closes.push(close);
        self.volumes.push(volume);
    }

    fn len(&self) -> usize {
        self.dates.len()
    }

    fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    indicators: Vec<IndicatorConfig>,
    long_entry_condition: String,
    long_exit_condition: String,
    short_entry_condition: String,
    short_exit_condition: String,
    take_profit_multiplier: f64,
    stop_loss_multiplier: f64,
    api_key: String,
    api_secret: String,
    symbol: String,
    interval: String,
    start_time: String,
    end_time: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct IndicatorConfig {
    name: String,
    parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Trade {
    entry_price: f64,
    exit_price: f64,
    entry_time: NaiveDateTime,
    exit_time: Option<NaiveDateTime>,
    profit_loss: f64,
    margin_used: f64,
    stop_loss: f64,
    take_profit: f64,
    is_long: bool,
    closed: bool,
}

pub struct Backtester {
    data: MarketData,
    indicators: HashMap<String, Vec<f64>>,
    trades: Vec<Trade>,
    initial_margin: f64,
    commission_rate: f64,
    equity: Vec<f64>,
    margin_percent_per_trade: f64,
    leverage: f64,
}

impl Backtester {
    pub fn new(data: MarketData, initial_margin: f64, commission_rate: f64, margin_percent_per_trade: f64, leverage: f64) -> Self {
        Self {
            data,
            indicators: HashMap::new(),
            trades: Vec::new(),
            initial_margin,
            commission_rate,
            equity: vec![initial_margin],
            margin_percent_per_trade,
            leverage,
        }
    }

    pub fn get_equity_curve(&self) -> &[f64] {
        &self.equity
    }

    pub fn get_trades_data(&self) -> &[Trade] {
        &self.trades
    }

    fn calculate_indicators(&mut self, configs: &[IndicatorConfig]) {
        configs.iter().for_each(|config| {
            let prices = &self.data.closes;
            let key = format!("{}_{}", config.name, config.parameters.get("period").unwrap_or(&0.0));

            let indicator_values = match config.name.as_str() {
                "sma" | "ema" => {
                    let period = config.parameters.get("period").copied().unwrap_or(20.0) as usize;
                    let period_u8: u8 = period.try_into().expect("Period value is out of u8 range");

                    match config.name.as_str() {
                        "sma" => {
                            let mut sma = SMA::new(period_u8, &prices[0]).unwrap();
                            prices.iter().map(|price| sma.next(price)).collect::<Vec<f64>>()
                        },
                        "ema" => {
                            let mut ema = EMA::new(period_u8, &prices[0]).unwrap();
                            prices.iter().map(|price| ema.next(price)).collect::<Vec<f64>>()
                        },
                        _ => unreachable!(),
                    }
                },
                _ => panic!("Unsupported indicator: {}", config.name),
            };
            self.indicators.insert(key, indicator_values);
        });
    }

    fn evaluate_complex_condition(condition: &str, indicators: &HashMap<String, Vec<f64>>, index: usize) -> bool {
        let parts: Vec<&str> = condition.split_whitespace().collect();
        match parts.len() {
            3 => {
                let left_expr = parts[0];
                let operator = parts[1];
                let right_expr = parts[2];

                let left_value = Self::evaluate_expression(left_expr, indicators, index);
                let right_value = Self::evaluate_expression(right_expr, indicators, index);

                match operator {
                    ">" => left_value > right_value,
                    "<" => left_value < right_value,
                    "=" => (left_value - right_value).abs() < f64::EPSILON,
                    _ => false,
                }
            },
            5 => {
                let left_condition = Self::evaluate_complex_condition(&parts[0..3].join(" "), indicators, index);
                let logical_operator = parts[3];
                let right_condition = Self::evaluate_complex_condition(&parts[4..].join(" "), indicators, index);

                match logical_operator {
                    "and" => left_condition && right_condition,
                    "or" => left_condition || right_condition,
                    _ => false,
                }
            },
            _ => false,
        }
    }

    fn evaluate_expression(expr: &str, indicators: &HashMap<String, Vec<f64>>, index: usize) -> f64 {
        if let Some((indicator_name, index_expr)) = expr.split_once('[') {
            let offset = index_expr.trim_end_matches(']').parse::<i32>().unwrap_or(0);
            let actual_index = (index as i32 + offset) as usize;
            indicators.get(indicator_name).and_then(|v| v.get(actual_index)).copied().unwrap_or(0.0)
        } else {
            indicators.get(expr).and_then(|v| v.get(index)).copied().unwrap_or(0.0)
        }
    }

    fn backtest(
        &mut self,
        long_entry_conditions: &[Box<dyn Fn(&Backtester, usize) -> bool + Send + Sync>],
        long_exit_conditions: &[Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>],
        short_entry_conditions: &[Box<dyn Fn(&Backtester, usize) -> bool + Send + Sync>],
        short_exit_conditions: &[Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>],
        take_profit_multiplier: f64,
        stop_loss_multiplier: f64,
    ) {
        let data = self.data.dates.clone();
        data.iter().enumerate().for_each(|(i, _)| {
            let current_equity = *self.equity.last().unwrap_or(&self.initial_margin);
            let margin_per_trade = current_equity * self.margin_percent_per_trade / 100.0;
            let entry_price = self.data.closes[i];
            let position_size = margin_per_trade * self.leverage / entry_price;

            if long_entry_conditions.iter().all(|condition| condition(self, i)) {
                self.open_long_trade(i, entry_price, position_size, take_profit_multiplier, stop_loss_multiplier);
            } else if short_entry_conditions.iter().all(|condition| condition(self, i)) {
                self.open_short_trade(i, entry_price, position_size, take_profit_multiplier, stop_loss_multiplier);
            }

            self.update_trades(i, long_exit_conditions, short_exit_conditions);
        });
    }

    #[inline]
    fn open_long_trade(&mut self, i: usize, entry_price: f64, position_size: f64, take_profit_multiplier: f64, stop_loss_multiplier: f64) {
        let entry_commission = entry_price * position_size * self.commission_rate / 100.0;
        let stop_loss = entry_price * (1.0 - stop_loss_multiplier / 100.0);
        let take_profit = entry_price * (1.0 + take_profit_multiplier / 100.0);

        if self.equity.last().unwrap_or(&self.initial_margin) > &(entry_commission + position_size) {
            let trade = Trade {
                entry_price,
                exit_price: 0.0,
                entry_time: self.data.dates[i],
                exit_time: None,
                profit_loss: -entry_commission,
                margin_used: position_size,
                stop_loss,
                take_profit,
                is_long: true,
                closed: false,
            };
            self.trades.push(trade);
            self.equity.push(self.equity.last().unwrap_or(&self.initial_margin) - (entry_commission + position_size));
        }
    }

    #[inline]
    fn open_short_trade(&mut self, i: usize, entry_price: f64, position_size: f64, take_profit_multiplier: f64, stop_loss_multiplier: f64) {
        let entry_commission = entry_price * position_size * self.commission_rate / 100.0;
        let stop_loss = entry_price * (1.0 + stop_loss_multiplier / 100.0);
        let take_profit = entry_price * (1.0 - take_profit_multiplier / 100.0);

        if self.equity.last().unwrap_or(&self.initial_margin) > &(entry_commission + position_size) {
            let trade = Trade {
                entry_price,
                exit_price: 0.0,
                entry_time: self.data.dates[i],
                exit_time: None,
                profit_loss: -entry_commission,
                margin_used: position_size,
                stop_loss,
                take_profit,
                is_long: false,
                closed: false,
            };
            self.trades.push(trade);
            self.equity.push(self.equity.last().unwrap_or(&self.initial_margin) - (entry_commission + position_size));
        }
    }

    #[inline]
    fn update_trades(
        &mut self,
        i: usize,
        long_exit_conditions: &[Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>],
        short_exit_conditions: &[Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>],
    ) {
        let highs = self.data.highs.clone();
        let lows = self.data.lows.clone();
        let dates = self.data.dates.clone();

        let mut new_equity = *self.equity.last().unwrap_or(&self.initial_margin);

        let mut to_close = Vec::new();

        // First pass: determine which trades to close
        for (j, trade) in self.trades.iter().enumerate() {
            if !trade.closed && dates[i] > trade.entry_time {
                let mut updated_trade = trade.clone();
                let exit_price;
                let exit_commission;
                let profit;

                if trade.is_long {
                    let condition_met = highs[i] >= trade.take_profit
                        || lows[i] <= trade.stop_loss
                        || long_exit_conditions.iter().any(|condition| condition(self, trade, i));
                    if condition_met {
                        exit_price = if lows[i] <= trade.stop_loss { trade.stop_loss } else { trade.take_profit };
                        exit_commission = exit_price * (trade.margin_used * self.leverage / trade.entry_price) * self.commission_rate / 100.0;
                        updated_trade.exit_price = exit_price;
                        updated_trade.exit_time = Some(dates[i]);
                        profit = (updated_trade.exit_price - trade.entry_price) * (trade.margin_used * self.leverage / trade.entry_price) - exit_commission;
                        updated_trade.profit_loss += profit;
                        new_equity += profit + trade.margin_used - exit_commission;
                        updated_trade.closed = true;
                        to_close.push((j, updated_trade));
                    }
                } else {
                    let condition_met = lows[i] <= trade.take_profit
                        || highs[i] >= trade.stop_loss
                        || short_exit_conditions.iter().any(|condition| condition(self, trade, i));
                    if condition_met {
                        exit_price = if highs[i] >= trade.stop_loss { trade.stop_loss } else { trade.take_profit };
                        exit_commission = exit_price * (trade.margin_used * self.leverage / trade.entry_price) * self.commission_rate / 100.0;
                        updated_trade.exit_price = exit_price;
                        updated_trade.exit_time = Some(dates[i]);
                        profit = (trade.entry_price - updated_trade.exit_price) * (trade.margin_used * self.leverage / trade.entry_price) - exit_commission;
                        updated_trade.profit_loss += profit;
                        new_equity += profit + trade.margin_used - exit_commission;
                        updated_trade.closed = true;
                        to_close.push((j, updated_trade));
                    }
                }
            }
        }

        // Second pass: close trades
        for (j, updated_trade) in to_close {
            self.trades[j] = updated_trade;
        }

        // Update equity
        self.equity.push(new_equity);
    }
}

pub fn fetch_binance_data(symbol: &str, interval: &str, start_time: &str, end_time: &str, api_key: &str) -> Result<MarketData, Error> {
    let client = Client::new();
    let url = format!(
        "https://api.binance.com/api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}",
        symbol, interval, start_time, end_time
    );

    let res = client
        .get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()?
        .json::<Value>()?;

    let mut data = MarketData::new(res.as_array().unwrap().len());

    for record in res.as_array().unwrap() {
        let date = NaiveDateTime::from_timestamp_millis(record[0].as_i64().unwrap()).unwrap();
        let open = record[1].as_str().unwrap().parse::<f64>().unwrap();
        let high = record[2].as_str().unwrap().parse::<f64>().unwrap();
        let low = record[3].as_str().unwrap().parse::<f64>().unwrap();
        let close = record[4].as_str().unwrap().parse::<f64>().unwrap();
        let volume = record[5].as_str().unwrap().parse::<f64>().unwrap();

        data.push(date, open, high, low, close, volume);
    }

    Ok(data)
}

pub fn backtesting(config: &str) -> Result<(Vec<f64>, Vec<Trade>), Box<dyn std::error::Error>> {
    let config: Config = serde_json::from_str(config)?;
    let data = fetch_binance_data(&config.symbol, &config.interval, &config.start_time, &config.end_time, &config.api_key)?;
    if data.is_empty() {
        return Err("Market data is empty".into());
    }

    let mut backtester = Backtester::new(data, 10000.0, 0.04, 10.0, 10.0);

    backtester.calculate_indicators(&config.indicators);

    let long_entry_conditions: Vec<Box<dyn Fn(&Backtester, usize) -> bool + Send + Sync>> = vec![
        Box::new(move |backtester, i| Backtester::evaluate_complex_condition(&config.long_entry_condition, &backtester.indicators, i)),
    ];
    let long_exit_conditions: Vec<Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>> = vec![
        Box::new(move |backtester, trade, i| Backtester::evaluate_complex_condition(&config.long_exit_condition, &backtester.indicators, i)),
    ];
    let short_entry_conditions: Vec<Box<dyn Fn(&Backtester, usize) -> bool + Send + Sync>> = vec![
        Box::new(move |backtester, i| Backtester::evaluate_complex_condition(&config.short_entry_condition, &backtester.indicators, i)),
    ];
    let short_exit_conditions: Vec<Box<dyn Fn(&Backtester, &Trade, usize) -> bool + Send + Sync>> = vec![
        Box::new(move |backtester, trade, i| Backtester::evaluate_complex_condition(&config.short_exit_condition, &backtester.indicators, i)),
    ];

    backtester.backtest(
        long_entry_conditions.as_slice(),
        long_exit_conditions.as_slice(),
        short_entry_conditions.as_slice(),
        short_exit_conditions.as_slice(),
        config.take_profit_multiplier,
        config.stop_loss_multiplier,
    );

    Ok((backtester.get_equity_curve().to_vec(), backtester.get_trades_data().to_vec()))
}

// fn main() {
//     let config = r#"
//     {
//         "indicators": [
//             {
//                 "name": "sma",
//                 "parameters": {
//                     "period": 50
//                 }
//             },
//             {
//                 "name": "ema",
//                 "parameters": {
//                     "period": 20
//                 }
//             }
//         ],
//         "long_entry_condition": "sma_50 < ema_20",
//         "long_exit_condition": "sma_50 > ema_20",
//         "short_entry_condition": "sma_50 > ema_20",
//         "short_exit_condition": "sma_50 < ema_20",
//         "take_profit_multiplier": 1.5,
//         "stop_loss_multiplier": 0.5,
//         "api_key": "YOUR_API_KEY",
//         "api_secret": "YOUR_API_SECRET",
//         "symbol": "BTCUSDT",
//         "interval": "1m",
//         "start_time": "1708114000000",
//         "end_time": "1715890000000"
//     }
//     "#;

//     match backtesting(config) {
//         Ok((equity_curve, trades)) => {
//             println!("Equity Curve: {:?}", equity_curve);
//             println!("Trades: {:?}", trades);
//         }
//         Err(e) => {
//             eprintln!("Error: {}", e);
//         }
//     }
// }
