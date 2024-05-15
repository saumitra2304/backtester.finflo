use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use csv::ReaderBuilder;
use std::collections::HashMap;
use yata::methods::{EMA, SMA};
use yata::prelude::*;

#[derive(Debug, Deserialize)]
struct CsvRecord {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

pub struct MarketData {
    dates: Vec<NaiveDateTime>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
}

impl MarketData {
    fn new() -> Self {
        Self {
            dates: Vec::new(),
            opens: Vec::new(),
            highs: Vec::new(),
            lows: Vec::new(),
            closes: Vec::new(),
            volumes: Vec::new(),
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
}

pub fn read_csv_data(path: &str) -> Result<MarketData, csv::Error> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut data = MarketData::new();

    for result in reader.deserialize() {
        let record: CsvRecord = result?;
        let date = NaiveDateTime::parse_from_str(&record.date, "%d-%m-%Y %H:%M").expect("Invalid date format");
        data.push(date, record.open, record.high, record.low, record.close, record.volume);
    }

    Ok(data)
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
        for config in configs {
            let prices = &self.data.closes; // Assume `closes` is Vec<f64>
            let key = format!("{}_{}", config.name, config.parameters.get("period").unwrap_or(&0.0));
    
            let indicator_values = match config.name.as_str() {
                "sma" | "ema" => {
                    let period = config.parameters.get("period")
                        .copied()
                        .unwrap_or(20.0) as usize; // Default period as usize
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
        }
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
        long_entry_conditions: &[Box<dyn Fn(&Backtester, usize) -> bool>],
        long_exit_conditions: &[Box<dyn Fn(&Backtester, &mut Trade, usize) -> bool>],
        short_entry_conditions: &[Box<dyn Fn(&Backtester, usize) -> bool>],
        short_exit_conditions: &[Box<dyn Fn(&Backtester, &mut Trade, usize) -> bool>],
        take_profit_multiplier: f64,
        stop_loss_multiplier: f64,
    ) {
        for i in 0..self.data.dates.len() {
            let current_equity = *self.equity.last().unwrap_or(&self.initial_margin);
            let margin_per_trade = current_equity * self.margin_percent_per_trade / 100.0;
            let entry_price = self.data.closes[i];
            let position_size = margin_per_trade * self.leverage / entry_price;

            if long_entry_conditions.iter().all(|condition| condition(self, i)) {
                self.open_long_trade(i, entry_price, position_size, take_profit_multiplier, stop_loss_multiplier);
            } else if short_entry_conditions.iter().all(|condition| condition(self, i)) {
                self.open_short_trade(i, entry_price, position_size, take_profit_multiplier, stop_loss_multiplier);
            }

            self.update_trades(i); // Update trades after conditions
        }
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
            };
            self.trades.push(trade);
            self.equity.push(self.equity.last().unwrap_or(&self.initial_margin) - (entry_commission + position_size));
        }
    }
    #[inline]
    fn update_trades(&mut self, i: usize) {
        let highs = &self.data.highs;
        let lows = &self.data.lows;
        let closes = &self.data.closes;
        let dates = &self.data.dates;

        for trade in &mut self.trades {
            if trade.exit_time.is_none() && dates[i] > trade.entry_time {
                if trade.is_long {
                    if highs[i] >= trade.take_profit || lows[i] <= trade.stop_loss {
                        let exit_price = if lows[i] <= trade.stop_loss { trade.stop_loss } else { trade.take_profit };
                        let exit_commission = exit_price * (trade.margin_used * self.leverage / trade.entry_price) * self.commission_rate / 100.0;
                        trade.exit_price = exit_price;
                        trade.exit_time = Some(dates[i]);
                        let profit = (trade.exit_price - trade.entry_price) * (trade.margin_used * self.leverage / trade.entry_price) - exit_commission;
                        trade.profit_loss += profit;
                        self.equity.push(self.equity.last().unwrap_or(&self.initial_margin) + profit + trade.margin_used - exit_commission);
                    }
                } else {
                    if lows[i] <= trade.take_profit || highs[i] >= trade.stop_loss {
                        let exit_price = if highs[i] >= trade.stop_loss { trade.stop_loss } else { trade.take_profit };
                        let exit_commission = exit_price * (trade.margin_used * self.leverage / trade.entry_price) * self.commission_rate / 100.0;
                        trade.exit_price = exit_price;
                        trade.exit_time = Some(dates[i]);
                        let profit = (trade.entry_price - trade.exit_price) * (trade.margin_used * self.leverage / trade.entry_price) - exit_commission;
                        trade.profit_loss += profit;
                        self.equity.push(self.equity.last().unwrap_or(&self.initial_margin) + profit + trade.margin_used - exit_commission);
                    }
                }
            }
        }
    }
}

pub fn backtesting(config: &str) -> Result<(Vec<f64>, Vec<Trade>), Box<dyn std::error::Error>> {
    let config: Config = serde_json::from_str(config)?;
    let data = read_csv_data("btc.csv")?;
    let mut backtester = Backtester::new(data, 10000.0, 0.04, 10.0, 10.0);

    backtester.calculate_indicators(&config.indicators);

    let long_entry_conditions: Vec<Box<dyn Fn(&Backtester, usize) -> bool>> = vec![
        Box::new(move |backtester, i| Backtester::evaluate_complex_condition(&config.long_entry_condition, &backtester.indicators, i)),
    ];
    let long_exit_conditions: Vec<Box<dyn Fn(&Backtester, &mut Trade, usize) -> bool>> = vec![
        Box::new(move |backtester, _trade, i| Backtester::evaluate_complex_condition(&config.long_exit_condition, &backtester.indicators, i)),
    ];
    let short_entry_conditions: Vec<Box<dyn Fn(&Backtester, usize) -> bool>> = vec![
        Box::new(move |backtester, i| Backtester::evaluate_complex_condition(&config.short_entry_condition, &backtester.indicators, i)),
    ];
    let short_exit_conditions: Vec<Box<dyn Fn(&Backtester, &mut Trade, usize) -> bool>> = vec![
        Box::new(move |backtester, _trade, i| Backtester::evaluate_complex_condition(&config.short_exit_condition, &backtester.indicators, i)),
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

fn main() {
    let config = r#"
    {
        "indicators": [
            {
                "name": "sma",
                "parameters": {
                    "period": 50
                }
            },
            {
                "name": "ema",
                "parameters": {
                    "period": 20
                }
            }
        ],
        "long_entry_condition": "sma_50 < ema_20",
        "long_exit_condition": "sma_50 > ema_20",
        "short_entry_condition": "sma_50 > ema_20",
        "short_exit_condition": "sma_50 < ema_20",
        "take_profit_multiplier": 1.5,
        "stop_loss_multiplier": 0.5
    }
    "#;

    match backtesting(config) {
        Ok((equity_curve, trades)) => {
            println!("Equity Curve: {:?}", equity_curve);
            println!("Trades: {:?}", trades);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
