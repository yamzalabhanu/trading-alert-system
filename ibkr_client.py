# ibkr_client.py
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---- dependency: ib-insync (add to requirements.txt) ----
# ib-insync wraps the official IB API with asyncio-friendly primitives
from ib_insync import (
    IB,
    Stock,
    Option,
    MarketOrder,
    LimitOrder,
    Contract,
    Trade,
)
# ---------------------------------------------------------

log = logging.getLogger("ibkr")
log.setLevel(logging.INFO)

# ---------- Configuration via env ----------
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
# Paper defaults: 7497 (TWS) or 4002 (Gateway). Override if needed.
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "123"))  # any stable int per app instance
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "")  # optional; leave empty to use default
IBKR_CONNECT_TIMEOUT = float(os.getenv("IBKR_CONNECT_TIMEOUT", "6.0"))
IBKR_RETRY = int(os.getenv("IBKR_RETRY", "3"))
IBKR_RETRY_DELAY = float(os.getenv("IBKR_RETRY_DELAY", "2.0"))

# Safety: default to paper-trading context; live trading requires explicit switch on TWS/IBG side.
IBKR_IS_PAPER = os.getenv("IBKR_IS_PAPER", "1") == "1"


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[int]
    status: str
    filled: float
    remaining: float
    avg_fill_price: Optional[float]
    error: Optional[str]
    raw: Optional[Dict[str, Any]]

    @staticmethod
    def from_trade(trade: Trade) -> "OrderResult":
        st = trade.orderStatus
        return OrderResult(
            ok=st.status in ("Submitted", "PreSubmitted", "Filled"),
            order_id=trade.order.orderId,
            status=st.status,
            filled=float(st.filled or 0.0),
            remaining=float(st.remaining or 0.0),
            avg_fill_price=float(st.avgFillPrice) if st.avgFillPrice not in (None, "") else None,
            error=None,
            raw={
                "order": trade.order.__dict__,
                "status": {
                    "status": st.status,
                    "filled": st.filled,
                    "remaining": st.remaining,
                    "avgFillPrice": st.avgFillPrice,
                    "permId": st.permId,
                    "clientId": st.clientId,
                },
                "log": [str(le) for le in trade.log],
            },
        )


class IBKRClient:
    """
    Thin async wrapper around ib-insync for placing orders from your app.
    Designed to be a singleton.
    """

    def __init__(
        self,
        host: str = IBKR_HOST,
        port: int = IBKR_PORT,
        client_id: int = IBKR_CLIENT_ID,
        account: str = IBKR_ACCOUNT,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.ib = IB()
        self._lock = asyncio.Lock()
        self._connected = False

    # ---------- connection ----------

    async def connect(self) -> None:
        """Connect with retries."""
        tries = IBKR_RETRY
        last_exc = None
        for i in range(tries):
            try:
                log.info(f"[IBKR] Connecting to {self.host}:{self.port} (clientId={self.client_id})...")
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=IBKR_CONNECT_TIMEOUT)
                self._connected = self.ib.isConnected()
                if self._connected:
                    log.info("[IBKR] Connected.")
                    return
            except Exception as e:
                last_exc = e
                log.warning(f"[IBKR] connect attempt {i+1}/{tries} failed: {e}")
                await asyncio.sleep(IBKR_RETRY_DELAY)
        raise RuntimeError(f"[IBKR] Failed to connect after {tries} attempts: {last_exc}")

    async def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        finally:
            self._connected = False
            log.info("[IBKR] Disconnected.")

    async def ensure_connected(self) -> None:
        if not self._connected or not self.ib.isConnected():
            async with self._lock:
                if not self._connected or not self.ib.isConnected():
                    await self.connect()

    # ---------- helpers ----------

    async def _qualify(self, contract: Contract) -> Contract:
        """Qualify a contract; IB requires fully qualified before placing orders."""
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            raise RuntimeError("Contract qualification failed; empty result.")
        return qualified[0]

    async def _await_first_status(self, trade: Trade, timeout: float = 10.0) -> OrderResult:
        """
        Wait briefly for the first orderStatus/log update so caller gets immediate feedback.
        """
        try:
            # wait for any of these to happen
            futs = [trade.orderStatusEvent, trade.statusEvent, trade.filledEvent, trade.cancelledEvent]
            await asyncio.wait(
                [asyncio.create_task(f) for f in futs],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:
            pass
        return OrderResult.from_trade(trade)

    # ---------- contract builders ----------

    @staticmethod
    def build_stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
        return Stock(symbol.upper(), exchange, currency)

    @staticmethod
    def build_option(
        symbol: str,
        right: str,             # "C" or "P" (case-insensitive)
        strike: float,
        expiry_yyyy_mm_dd: str, # "YYYY-MM-DD"
        exchange: str = "SMART",
        currency: str = "USD",
        multiplier: str = "100",
    ) -> Option:
        uid = expiry_yyyy_mm_dd.replace("-", "")  # "YYYYMMDD"
        return Option(
            symbol.upper(),
            uid,
            float(strike),
            right.upper()[0],
            exchange,
            currency=currency,
            multiplier=multiplier,
        )

    @staticmethod
    def side_to_action(side: str) -> str:
        """Maps your 'CALL/PUT' + decision to BUY/SELL action; typically you BUY when LLM says buy."""
        s = side.strip().upper()
        if s in ("CALL", "C", "BUY_CALL", "BUY"):
            return "BUY"
        if s in ("PUT", "P", "BUY_PUT", "SELL"):  # caller can override explicitly
            # NOTE: we cannot infer SELL vs BUY solely from 'PUT'.
            # Keep default action 'BUY' for buys; pass 'SELL' explicitly if you mean to sell.
            return "BUY"
        return "BUY"

    # ---------- order APIs (stocks) ----------

    async def place_stock_market(
        self, symbol: str, quantity: int, action: str = "BUY", tif: str = "DAY", account: Optional[str] = None
    ) -> OrderResult:
        await self.ensure_connected()
        contract = await self._qualify(self.build_stock(symbol))
        order = MarketOrder(action.upper(), int(quantity), tif=tif.upper())
        if account or self.account:
            order.account = account or self.account
        trade = self.ib.placeOrder(contract, order)
        return await self._await_first_status(trade)

    async def place_stock_limit(
        self, symbol: str, quantity: int, limit_price: float, action: str = "BUY", tif: str = "DAY", account: Optional[str] = None
    ) -> OrderResult:
        await self.ensure_connected()
        contract = await self._qualify(self.build_stock(symbol))
        order = LimitOrder(action.upper(), int(quantity), float(limit_price), tif=tif.upper())
        if account or self.account:
            order.account = account or self.account
        trade = self.ib.placeOrder(contract, order)
        return await self._await_first_status(trade)

    # ---------- order APIs (options) ----------

    async def place_option_market(
        self,
        symbol: str,
        right: str,               # "C" or "P"
        strike: float,
        expiry_yyyy_mm_dd: str,   # "YYYY-MM-DD"
        quantity: int,
        action: str = "BUY",
        tif: str = "DAY",
        account: Optional[str] = None,
    ) -> OrderResult:
        await self.ensure_connected()
        contract = await self._qualify(self.build_option(symbol, right, strike, expiry_yyyy_mm_dd))
        order = MarketOrder(action.upper(), int(quantity), tif=tif.upper())
        if account or self.account:
            order.account = account or self.account
        trade = self.ib.placeOrder(contract, order)
        return await self._await_first_status(trade)

    async def place_option_limit(
        self,
        symbol: str,
        right: str,
        strike: float,
        expiry_yyyy_mm_dd: str,
        quantity: int,
        limit_price: float,
        action: str = "BUY",
        tif: str = "DAY",
        account: Optional[str] = None,
    ) -> OrderResult:
        await self.ensure_connected()
        contract = await self._qualify(self.build_option(symbol, right, strike, expiry_yyyy_mm_dd))
        order = LimitOrder(action.upper(), int(quantity), float(limit_price), tif=tif.upper())
        if account or self.account:
            order.account = account or self.account
        trade = self.ib.placeOrder(contract, order)
        return await self._await_first_status(trade)

    # ---------- bracket helpers ----------

    async def place_stock_bracket_limit(
        self,
        symbol: str,
        quantity: int,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        action: str = "BUY",
        tif: str = "DAY",
        account: Optional[str] = None,
    ) -> List[OrderResult]:
        """
        Places a 3-legged bracket: parent (limit), take-profit (limit), stop-loss (stop).
        Returns results for each leg (parent first).
        """
        await self.ensure_connected()
        contract = await self._qualify(self.build_stock(symbol))

        # Build bracket orders manually for clarity
        parent = LimitOrder(action.upper(), int(quantity), float(limit_price), tif=tif.upper())
        tp = LimitOrder("SELL" if action.upper() == "BUY" else "BUY", int(quantity), float(take_profit_price), tif=tif.upper())
        sl = MarketOrder("SELL" if action.upper() == "BUY" else "BUY", int(quantity), tif=tif.upper())
        sl.auxPrice = float(stop_loss_price)  # stop trigger

        for o in (parent, tp, sl):
            if account or self.account:
                o.account = account or self.account
            o.transmit = False

        trade_parent = self.ib.placeOrder(contract, parent)
        tp.parentId = trade_parent.order.orderId
        trade_tp = self.ib.placeOrder(contract, tp)
        sl.parentId = trade_parent.order.orderId
        sl.orderType = "STP"
        trade_sl = self.ib.placeOrder(contract, sl)

        # Transmit the last leg to send all
        sl.transmit = True
        self.ib.placeOrder(contract, sl)

        res_parent = await self._await_first_status(trade_parent)
        res_tp = await self._await_first_status(trade_tp)
        res_sl = await self._await_first_status(trade_sl)
        return [res_parent, res_tp, res_sl]

    # ---------- admin ----------

    async def cancel(self, order_id: int) -> None:
        await self.ensure_connected()
        self.ib.cancelOrder(order_id)

    async def open_orders(self) -> List[Dict[str, Any]]:
        await self.ensure_connected()
        orders = self.ib.reqOpenOrders()
        out = []
        for tr in orders:
            st = tr.orderStatus
            out.append(
                {
                    "orderId": tr.order.orderId,
                    "status": st.status,
                    "filled": st.filled,
                    "remaining": st.remaining,
                    "avgFillPrice": st.avgFillPrice,
                    "symbol": getattr(tr.contract, "symbol", None),
                    "secType": getattr(tr.contract, "secType", None),
                    "action": tr.order.action,
                    "lmtPrice": getattr(tr.order, "lmtPrice", None),
                }
            )
        return out

    async def positions(self) -> List[Dict[str, Any]]:
        await self.ensure_connected()
        ps = await self.ib.reqPositionsAsync()
        out = []
        for acc, con, pos, avgCost in ps:
            out.append(
                {
                    "account": acc,
                    "symbol": getattr(con, "symbol", None),
                    "secType": getattr(con, "secType", None),
                    "position": pos,
                    "avgCost": avgCost,
                }
            )
        return out


# Singleton you can import elsewhere
ibkr = IBKRClient()

# ---------- Convenience functions tailored to your app ----------

async def place_recommended_option_order(
    *,
    symbol: str,
    side: str,                   # "CALL" or "PUT"
    strike: float,
    expiry_iso: str,             # "YYYY-MM-DD"
    quantity: int = 1,
    limit_price: Optional[float] = None,
    action: str = "BUY",         # override if you need SELL
    tif: str = "DAY",
    account: Optional[str] = None,
) -> OrderResult:
    """
    High-level helper: place an option order using your alert fields.
    If limit_price is None -> market order; else limit order.
    """
    right = "C" if side.upper().startswith("C") else "P"
    if limit_price is None:
        return await ibkr.place_option_market(
            symbol=symbol, right=right, strike=strike, expiry_yyyy_mm_dd=expiry_iso,
            quantity=quantity, action=action, tif=tif, account=account
        )
    else:
        return await ibkr.place_option_limit(
            symbol=symbol, right=right, strike=strike, expiry_yyyy_mm_dd=expiry_iso,
            quantity=quantity, limit_price=limit_price, action=action, tif=tif, account=account
        )
