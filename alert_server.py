# main.py with VWAP and Bollinger Bands enhancements
# [Truncated for brevity — full logic will include:]
# - VWAP/BB fetch function
# - BB breakout detector
# - Integration into /webhook
# - Prompt enhancement for GPT
# Full version built on top of the provided base

# Enhanced Trading System
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional
import httpx

# ======================
# Enhanced Technical Analysis
# ======================

async def get_advanced_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Fetch comprehensive technical indicators"""
    return {
        "rsi": await calculate_rsi(symbol),
        "macd": await calculate_macd(symbol),
        "vwap_bb": await get_vwap_and_bollinger(symbol),
        "volume_profile": await get_volume_profile(symbol),
        "multi_timeframe": await multi_timeframe_analysis(symbol)
    }

async def calculate_rsi(symbol: str, period: int = 14) -> float:
    """Calculate RSI using Polygon API"""
    url = (f"https://api.polygon.io/v1/indicators/rsi/{symbol}?"
           f"timespan=minute&window={period}&series_type=close&apiKey={POLYGON_API_KEY}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            return data["results"]["values"][0]["value"]
    except Exception:
        logging.warning("RSI calculation failed")
        return 0.0

async def calculate_macd(symbol: str) -> Dict[str, float]:
    """Calculate MACD values"""
    url = (f"https://api.polygon.io/v1/indicators/macd/{symbol}?"
           f"timespan=minute&short_window=12&long_window=26&signal_window=9&apiKey={POLYGON_API_KEY}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            values = data["results"]["values"][0]
            return {
                "macd": values["value"],
                "signal": values["signal"],
                "histogram": values["histogram"]
            }
    except Exception:
        logging.warning("MACD calculation failed")
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

async def get_volume_profile(symbol: str) -> Dict[str, float]:
    """Identify significant volume nodes"""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            return {
                "vpoc": data["ticker"]["min"]["p"],  # Volume Point of Control
                "high_volume_nodes": data["ticker"]["prevDay"]["h"]
            }
    except Exception:
        logging.warning("Volume profile fetch failed")
        return {"vpoc": 0.0, "high_volume_nodes": []}

# ======================
# Multi-Timeframe Analysis
# ======================

async def multi_timeframe_analysis(symbol: str) -> Dict[str, str]:
    """Check trendline breakouts across timeframes"""
    return {
        "5min": await detect_trendline_breakout(symbol, "5minute"),
        "15min": await detect_trendline_breakout(symbol, "15minute"),
        "1hr": await detect_trendline_breakout(symbol, "1hour")
    }

async def detect_trendline_breakout(symbol: str, timespan: str) -> str:
    """Enhanced breakout detection with variable timeframes"""
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=3 if "hour" in timespan else 1)
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/"
               f"{start.date()}/{end.date()}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}")
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json().get("results", [])
        
        if len(data) < 20:
            return "insufficient data"

        closes = np.array([bar["c"] for bar in data])
        x = np.arange(len(closes))
        
        # Fit linear trendline
        p = Polynomial.fit(x, closes, 1)
        trend = p(x)
        std_dev = np.std(closes - trend)
        
        # Check breakout conditions
        latest = closes[-1]
        upper_bound = trend[-1] + std_dev
        lower_bound = trend[-1] - std_dev

        if latest > upper_bound:
            return "breakout"
        elif latest < lower_bound:
            return "breakdown"
        return "neutral"
    except Exception:
        return "error"

# ======================
# Market Context Integration
# ======================

async def get_market_context() -> Dict[str, Any]:
    """Fetch comprehensive market context"""
    return {
        "vix": await get_vix_value(),
        "sector_performance": await get_sector_performance(),
        "market_breadth": await get_market_breadth(),
        "volatility_regime": await get_volatility_regime()
    }

async def get_vix_value() -> float:
    """Get current VIX value"""
    url = "https://api.polygon.io/v2/aggs/ticker/VIX/range/1/minute/last?adjusted=true&apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url.format(apiKey=POLYGON_API_KEY))
            return response.json()["results"][0]["c"]
    except Exception:
        return 0.0

async def get_sector_performance() -> Dict[str, float]:
    """Get real-time sector performance"""
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/sectors?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url.format(apiKey=POLYGON_API_KEY))
            return {sector["sector"]: sector["performance"] for sector in response.json()["sectors"]}
    except Exception:
        return {}

async def get_market_breadth() -> Dict[str, float]:
    """Calculate market breadth metrics"""
    url = ("https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url.format(apiKey=POLYGON_API_KEY))
            return {
                "advancers": response.json()["exchanges"]["nyse"]["advancers"],
                "decliners": response.json()["exchanges"]["nyse"]["decliners"]
            }
    except Exception:
        return {"advancers": 0, "decliners": 0}

# ======================
# Liquidity Analysis
# ======================

async def assess_liquidity(symbol: str) -> Dict[str, Any]:
    """Evaluate market liquidity conditions"""
    return {
        "order_book": await get_order_book_depth(symbol),
        "spread_analysis": await calculate_spread_analysis(symbol),
        "slippage_estimate": await estimate_slippage(symbol)
    }

async def get_order_book_depth(symbol: str) -> Dict[str, List[float]]:
    """Get order book depth"""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            book = response.json()["ticker"]["lastQuote"]
            return {
                "bids": [book["bp"], book["bp"] - 0.01],  # Sample bid levels
                "asks": [book["ap"], book["ap"] + 0.01]   # Sample ask levels
            }
    except Exception:
        return {"bids": [], "asks": []}

async def calculate_spread_analysis(symbol: str) -> float:
    """Calculate current spread percentage"""
    url = f"https://api.polygon.io/v1/last_quote/stocks/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            quote = response.json()["last"]
            spread = quote["askprice"] - quote["bidprice"]
            return (spread / quote["askprice"]) * 100
    except Exception:
        return 0.0

async def estimate_slippage(symbol: str, quantity: int = 100) -> float:
    """Estimate slippage for a standard order"""
    spread = await calculate_spread_analysis(symbol)
    return spread * 0.5 * (quantity / 100)  # Simplified model

# ======================
# Backtesting Engine
# ======================

async def backtest_strategy(symbol: str, strategy: str) -> Dict[str, float]:
    """Backtest strategy performance"""
    data = await get_historical_data(symbol)
    signals = generate_signals(data, strategy)
    results = calculate_performance(data, signals)
    return results

async def get_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch historical data for backtesting"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
           f"{start.date()}/{end.date()}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        bars = response.json().get("results", [])
    
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("date")

def generate_signals(data: pd.DataFrame, strategy: str) -> pd.Series:
    """Generate trading signals based on strategy"""
    if strategy == "trend_following":
        data["sma20"] = data["c"].rolling(20).mean()
        data["sma50"] = data["c"].rolling(50).mean()
        return np.where(data["sma20"] > data["sma50"], 1, 0)
    elif strategy == "mean_reversion":
        data["rsi"] = calculate_rsi_series(data["c"])
        return np.where(data["rsi"] < 30, 1, np.where(data["rsi"] > 70, -1, 0))
    return pd.Series(0, index=data.index)

def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a price series"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_performance(data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics"""
    data["position"] = signals.shift()
    data["returns"] = data["c"].pct_change()
    data["strategy"] = data["position"] * data["returns"]
    
    return {
        "total_return": data["strategy"].sum() * 100,
        "win_rate": (data["strategy"] > 0).mean() * 100,
        "max_drawdown": (data["strategy"].cumsum() - data["strategy"].cumsum().cummax()).min() * 100
    }

# ======================
# Live Market Monitoring
# ======================

class MarketConditionMonitor:
    """Real-time market condition tracker"""
    def __init__(self):
        self.conditions = {
            "trend_strength": "neutral",
            "volatility_regime": "normal",
            "market_breadth": "neutral"
        }
        self.last_update = datetime.min
        
    async def update(self):
        """Update market conditions"""
        if datetime.utcnow() - self.last_update < timedelta(minutes=5):
            return
            
        try:
            breadth = await get_market_breadth()
            vix = await get_vix_value()
            
            # Update trend strength
            sp500_trend = await detect_trendline_breakout("SPY", "1day")
            self.conditions["trend_strength"] = (
                "strong_up" if sp500_trend == "breakout" else
                "strong_down" if sp500_trend == "breakdown" else "neutral"
            )
            
            # Update volatility regime
            self.conditions["volatility_regime"] = (
                "high" if vix > 30 else
                "low" if vix < 15 else "normal"
            )
            
            # Update market breadth
            adv_ratio = breadth["advancers"] / (breadth["advancers"] + breadth["decliners"])
            self.conditions["market_breadth"] = (
                "positive" if adv_ratio > 0.6 else
                "negative" if adv_ratio < 0.4 else "neutral"
            )
            
            self.last_update = datetime.utcnow()
        except Exception as e:
            logging.error(f"Market condition update failed: {e}")

async def get_volatility_regime() -> str:
    """Classify current volatility regime"""
    vix = await get_vix_value()
    if vix > 30:
        return "high_volatility"
    elif vix < 15:
        return "low_volatility"
    return "normal_volatility"

# ======================
# Enhanced GPT Analysis
# ======================

async def generate_gpt_analysis(symbol: str, alert: Alert) -> str:
    """Generate enhanced trading analysis with all features"""
    market_context = await get_market_context()
    technicals = await get_advanced_technical_indicators(symbol)
    liquidity = await assess_liquidity(symbol)
    backtest = await backtest_strategy(symbol, "trend_following")
    
    prompt = f"""
**Market Context**:
- VIX: {market_context['vix']:.2f}
- Top Sector: {max(market_context['sector_performance'], key=market_context['sector_performance'].get)} 
- Market Breadth: {market_context['market_breadth']['advancers']}/{market_context['market_breadth']['decliners']}
- Volatility Regime: {market_context['volatility_regime']}

**Technical Analysis for {symbol}**:
- RSI(14): {technicals['rsi']:.2f}
- MACD: {technicals['macd']['macd']:.4f} (Signal: {technicals['macd']['signal']:.4f})
- Multi-timeframe Trend:
  • 5min: {technicals['multi_timeframe']['5min']}
  • 15min: {technicals['multi_timeframe']['15min']}
  • 1hr: {technicals['multi_timeframe']['1hr']}
- Volume Profile: VPOC @ {technicals['volume_profile']['vpoc']:.2f}

**Liquidity Analysis**:
- Spread: {liquidity['spread_analysis']:.2f}%
- Estimated Slippage: {liquidity['slippage_estimate']:.4f}
- Order Book Depth: {len(liquidity['order_book']['bids'])} levels

**Backtest Results (30-day trend following)**:
- Total Return: {backtest['total_return']:.2f}%
- Win Rate: {backtest['win_rate']:.2f}%
- Max Drawdown: {backtest['max_drawdown']:.2f}%

**Trade Signal**:
- Symbol: {symbol}
- Signal: {alert.signal.upper()}
- Trigger Price: {alert.price:.2f}

**Recommendation Format**:
- Decision: [Buy/Pass]
- Confidence: [0-100]
- Position Size: [% of portfolio]
- Risk: [Low/Medium/High]
- Timeframe: [Intraday/Swing]
- Reasoning: [1-2 sentences]
"""
    return await get_gpt_response(prompt)

# ======================
# Webhook Integration
# ======================

@app.post("/enhanced-webhook")
async def handle_enhanced_alert(alert: Alert):
    """Process alerts with all enhancements"""
    # Update market conditions
    monitor = MarketConditionMonitor()
    await monitor.update()
    
    # Fetch all analytical data
    market_context = await get_market_context()
    technicals = await get_advanced_technical_indicators(alert.symbol)
    liquidity = await assess_liquidity(alert.symbol)
    backtest = await backtest_strategy(alert.symbol, "trend_following")
    
    # Generate GPT analysis
    gpt_reply = await generate_gpt_analysis(alert.symbol, alert)
    
    # Compose Telegram message
    message = (
        f"🚀 *Enhanced Trade Alert*: {alert.symbol} {alert.signal.upper()} @ ${alert.price:.2f}\n"
        f"📊 *Market Context*: {monitor.conditions['trend_strength']} | {monitor.conditions['volatility_regime']}\n"
        f"💡 *GPT Analysis*:\n{gpt_reply}"
    )
    await send_telegram_message(message)
    
    # Update cooldown and logs
    update_cooldown(alert.symbol, alert.signal)
    log_signal(alert.symbol, alert.signal, gpt_reply)
    
    return {"status": "processed", "analysis": gpt_reply}

# Initialize market monitor
market_monitor = MarketConditionMonitor()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(market_monitor.update())
    asyncio.create_task(schedule_daily_summary())
    asyncio.create_task(scan_unusual_activity())
