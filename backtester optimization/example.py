import requests
import json

url = "http://127.0.0.1:8080/backtest"
data = {
    "indicators": [
        {"name": "sma", "parameters": {"period": 50}},
        {"name": "ema", "parameters": {"period": 20}}
    ],
    "long_entry_condition": "sma_50 < ema_20",
    "long_exit_condition": "sma_50 > ema_20",
    "short_entry_condition": "sma_50 > ema_20",
    "short_exit_condition": "sma_50 < ema_20",
    "take_profit_multiplier": 1.5,
    "stop_loss_multiplier": 0.5
}

response = requests.post(url, json=data)
print(response.json())
