import requests
import logging
import time
import MySQLdb
import json
import websocket
import pandas as pd
import pandas_ta as ta

from core.config import config
from datetime import datetime, timedelta
from core.models import IntraDayData

log = logging.getLogger(__name__)

ALPACA_API_KEY = config.alpaca.apikey
ALPACA_SECRET_KEY = config.alpaca.secret

ALPACA_WEBSOCKET_URL = config.alpaca.stockSocketUrl

def on_open(ws):
    log.debug("WebSocket connection opened.")
    auth_message = {
        "action": "auth",
        "key": ALPACA_API_KEY,
        "secret": ALPACA_SECRET_KEY
    }
    ws.send(json.dumps(auth_message))
    
    subscribe_message = {
        "action": "subscribe",
        "trades": ["AAPL"],    
        "quotes": ["AAPL"],
        "bars": ["AAPL"]
    }
    ws.send(json.dumps(subscribe_message))

def on_message(ws, message):
    log.debug(f"Received message: {message}")

def on_error(ws, error):
    log.debug(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    log.debug("WebSocket connection closed.")

if __name__ == "__main__":

    ws = websocket.WebSocketApp(
        ALPACA_WEBSOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
