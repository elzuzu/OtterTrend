# Phase 1 - Market & Portfolio (MEXC via CCXT)

> **Objectif**: Implémenter l'interface avec l'exchange MEXC pour récupérer les données marché, gérer le portefeuille et exécuter des ordres.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (structure, config, memory)

---

## T1.1 - Interface Exchange Générique

### T1.1.1 - Créer le wrapper CCXT multi-exchange
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `src/tools/market.py` avec support pour MEXC (principal), Bybit et OKX (backup).

**Architecture** :
```python
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
from src.config import get_config

class ExchangeClient:
    """Interface unifiée pour les exchanges via CCXT"""

    def __init__(self) -> None:
        self._exchange: Optional[ccxt.Exchange] = None
        self._config = get_config()

    def _create_exchange(self) -> ccxt.Exchange:
        """Factory pour créer l'instance exchange selon config"""
        exchange_id = self._config.exchange_id
        params = {
            "apiKey": self._config.exchange_api_key,
            "secret": self._config.exchange_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        # Spécificités par exchange
        if exchange_id == "mexc":
            params["options"]["recvWindow"] = 60000
        elif exchange_id == "okx":
            params["password"] = os.getenv("EXCHANGE_PASSWORD", "")

        cls = getattr(ccxt, exchange_id)
        exchange = cls(params)

        if self._config.exchange_testnet:
            exchange.set_sandbox_mode(True)

        return exchange

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = self._create_exchange()
        return self._exchange

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
```

**Critères de validation**:
- [ ] Support MEXC, Bybit, OKX via variable d'env
- [ ] Mode sandbox/testnet configurable
- [ ] Rate limiting activé par défaut
- [ ] Connexion lazy (créée au premier appel)
- [ ] Méthode close() pour cleanup

---

### T1.1.2 - Implémenter les méthodes de lecture marché
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Ajouter les méthodes suivantes à `ExchangeClient` :

```python
async def get_market_price(self, symbol: str) -> float:
    """Récupère le dernier prix pour un symbole"""
    ticker = await self.exchange.fetch_ticker(symbol)
    return float(ticker["last"])

async def get_ticker(self, symbol: str) -> Dict[str, Any]:
    """Récupère le ticker complet"""
    ticker = await self.exchange.fetch_ticker(symbol)
    return {
        "symbol": symbol,
        "last": float(ticker["last"]),
        "bid": float(ticker.get("bid", 0)),
        "ask": float(ticker.get("ask", 0)),
        "volume_24h": float(ticker.get("baseVolume", 0)),
        "quote_volume_24h": float(ticker.get("quoteVolume", 0)),
        "change_24h_pct": float(ticker.get("percentage", 0)),
        "high_24h": float(ticker.get("high", 0)),
        "low_24h": float(ticker.get("low", 0)),
    }

async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
    """Récupère les tickers pour plusieurs symboles"""
    tickers = await self.exchange.fetch_tickers(symbols)
    result = {}
    for sym, data in tickers.items():
        result[sym] = {
            "last": float(data["last"]),
            "volume_24h": float(data.get("baseVolume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
            "change_24h_pct": float(data.get("percentage", 0)),
        }
    return result

async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
    """Récupère le carnet d'ordres"""
    ob = await self.exchange.fetch_order_book(symbol, limit)
    return {
        "symbol": symbol,
        "bids": ob["bids"][:limit],  # [[price, amount], ...]
        "asks": ob["asks"][:limit],
        "timestamp": ob.get("timestamp"),
    }

async def get_ohlcv(
    self, symbol: str, timeframe: str = "1h", limit: int = 100
) -> List[Dict]:
    """Récupère les chandeliers OHLCV"""
    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return [
        {
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
        }
        for candle in ohlcv
    ]
```

**Critères de validation**:
- [ ] Toutes les méthodes async
- [ ] Données normalisées (pas de None, floats propres)
- [ ] Gestion des erreurs CCXT
- [ ] Tests avec mock exchange

---

## T1.2 - Gestion du Portefeuille

### T1.2.1 - Implémenter les méthodes de lecture portefeuille
**Priorité**: CRITIQUE
**Estimation**: Moyenne

```python
async def get_balance(self) -> Dict[str, Any]:
    """Récupère la balance complète du compte"""
    balance = await self.exchange.fetch_balance()
    return {
        "total": {k: float(v) for k, v in balance.get("total", {}).items() if v > 0},
        "free": {k: float(v) for k, v in balance.get("free", {}).items() if v > 0},
        "used": {k: float(v) for k, v in balance.get("used", {}).items() if v > 0},
    }

async def get_base_balance(self) -> float:
    """Récupère la balance en devise de base (USDT)"""
    cfg = get_config()
    balance = await self.get_balance()
    return balance["free"].get(cfg.base_currency, 0.0)

async def get_portfolio_state(self) -> Dict[str, Any]:
    """
    État complet du portefeuille avec valorisation
    Retourne:
    - balance_usdt: solde libre en USDT
    - total_equity_usdt: valeur totale du portefeuille
    - positions: liste des positions avec valorisation
    """
    cfg = get_config()
    balance = await self.get_balance()
    base_bal = balance["free"].get(cfg.base_currency, 0.0)

    positions = []
    total_equity = base_bal

    for asset, qty in balance["total"].items():
        if asset.upper() == cfg.base_currency.upper():
            continue
        if qty <= 0:
            continue

        symbol = f"{asset}/{cfg.base_currency}"
        try:
            price = await self.get_market_price(symbol)
            value_usdt = qty * price
            total_equity += value_usdt
            positions.append({
                "asset": asset,
                "symbol": symbol,
                "amount": qty,
                "price": price,
                "value_usdt": value_usdt,
            })
        except Exception:
            # Pair non tradeable ou erreur
            continue

    return {
        "mode": "paper" if cfg.paper_trading else "live",
        "exchange": cfg.exchange_id,
        "balance_usdt": base_bal,
        "total_equity_usdt": total_equity,
        "positions": positions,
        "positions_count": len(positions),
    }
```

**Critères de validation**:
- [ ] Balance en temps réel
- [ ] Valorisation des positions en USDT
- [ ] Calcul d'equity total
- [ ] Gestion des assets non-tradeables

---

### T1.2.2 - Implémenter le snapshot marché complet
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def get_market_snapshot(
    self,
    symbols: Optional[List[str]] = None,
    include_analytics: bool = True,
) -> Dict[str, Any]:
    """
    Snapshot complet pour le LLM:
    - Tickers des symboles suivis
    - Portfolio state
    - Optionnel: analytics (regime, volatilité)
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    tickers = await self.get_tickers(symbols)
    portfolio = await self.get_portfolio_state()

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": self._config.exchange_id,
        "tickers": tickers,
        "portfolio": portfolio,
    }

    if include_analytics:
        from src.tools.analytics import compute_market_analytics
        # Récupérer OHLCV pour analytics
        for sym in symbols[:3]:  # Limiter pour perf
            try:
                ohlcv = await self.get_ohlcv(sym, "1h", 48)
                prices = [c["close"] for c in ohlcv]
                analytics = compute_market_analytics(prices)
                snapshot.setdefault("analytics", {})[sym] = analytics
            except Exception:
                continue

    return snapshot
```

**Critères de validation**:
- [ ] Snapshot structuré et complet
- [ ] Timestamp UTC
- [ ] Analytics optionnels
- [ ] Performance acceptable (< 5s)

---

## T1.3 - Exécution d'Ordres

### T1.3.1 - Implémenter la passation d'ordres
**Priorité**: CRITIQUE
**Estimation**: Haute

```python
async def place_order(
    self,
    symbol: str,
    side: str,  # 'buy' | 'sell'
    amount: float,
    order_type: str = "market",
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Place un ordre sur l'exchange.
    IMPORTANT: Le risk check doit être fait AVANT d'appeler cette méthode!
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"Side invalide: {side}")

    if order_type == "market":
        order = await self.exchange.create_market_order(symbol, side, amount)
    elif order_type == "limit":
        if price is None:
            raise ValueError("Prix requis pour ordre limit")
        order = await self.exchange.create_limit_order(symbol, side, amount, price)
    else:
        raise ValueError(f"Type d'ordre non supporté: {order_type}")

    return {
        "id": order["id"],
        "symbol": order["symbol"],
        "side": order["side"],
        "type": order["type"],
        "amount": float(order["amount"]),
        "price": float(order.get("average") or order.get("price") or 0),
        "cost": float(order.get("cost", 0)),
        "status": order["status"],
        "timestamp": order.get("timestamp"),
        "fee": order.get("fee"),
    }

async def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
    """Ferme toute la position sur un symbole (vend tout)"""
    base_asset = symbol.split("/")[0]
    balance = await self.get_balance()
    qty = balance["free"].get(base_asset, 0.0)

    if qty <= 0:
        return None

    # Vérifier la taille minimum d'ordre
    try:
        markets = await self.exchange.load_markets()
        market = markets.get(symbol)
        if market:
            min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
            if qty < min_amount:
                return None
    except Exception:
        pass

    return await self.place_order(symbol, "sell", qty, "market")

async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
    """Récupère le statut d'un ordre"""
    order = await self.exchange.fetch_order(order_id, symbol)
    return {
        "id": order["id"],
        "symbol": order["symbol"],
        "status": order["status"],
        "filled": float(order.get("filled", 0)),
        "remaining": float(order.get("remaining", 0)),
        "average_price": float(order.get("average") or 0),
    }

async def cancel_order(self, order_id: str, symbol: str) -> bool:
    """Annule un ordre ouvert"""
    try:
        await self.exchange.cancel_order(order_id, symbol)
        return True
    except Exception:
        return False
```

**Critères de validation**:
- [ ] Support ordres market et limit
- [ ] Retour normalisé avec toutes les infos
- [ ] Gestion des minimums d'ordre
- [ ] close_position vend la totalité
- [ ] Gestion des erreurs (balance insuffisante, etc.)

---

## T1.4 - Intelligence Exchange (MEXC Spécifique)

### T1.4.1 - Implémenter la détection de nouveaux listings
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def get_all_markets(self) -> List[Dict[str, Any]]:
    """Récupère tous les marchés disponibles"""
    markets = await self.exchange.load_markets(reload=True)
    result = []
    for symbol, market in markets.items():
        if market.get("spot") and market.get("active"):
            result.append({
                "symbol": symbol,
                "base": market["base"],
                "quote": market["quote"],
                "active": market["active"],
            })
    return result

async def detect_new_listings(
    self,
    known_symbols: Optional[set] = None,
    quote_filter: str = "USDT",
) -> List[Dict[str, Any]]:
    """
    Détecte les nouveaux listings depuis la dernière vérification.
    Compare avec les symboles connus en mémoire/DB.
    """
    markets = await self.get_all_markets()

    current_symbols = {
        m["symbol"] for m in markets
        if m["quote"] == quote_filter
    }

    if known_symbols is None:
        # Premier appel - tout est "nouveau"
        return []

    new_symbols = current_symbols - known_symbols
    new_listings = []

    for sym in new_symbols:
        try:
            ticker = await self.get_ticker(sym)
            new_listings.append({
                "symbol": sym,
                "is_new": True,
                "price": ticker["last"],
                "volume_24h": ticker["volume_24h"],
                "discovered_at": datetime.utcnow().isoformat(),
            })
        except Exception:
            continue

    return new_listings
```

**Critères de validation**:
- [ ] Détection efficace des nouveaux tokens
- [ ] Filtrage par quote currency
- [ ] Enrichissement avec ticker data

---

### T1.4.2 - Implémenter le scan des top gainers
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def get_top_gainers(
    self,
    quote_filter: str = "USDT",
    min_volume_usdt: float = 10000,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Récupère les top gainers 24h.
    Filtré par volume minimum pour éviter les illiquides.
    """
    try:
        tickers = await self.exchange.fetch_tickers()
    except Exception:
        return []

    gainers = []

    for symbol, data in tickers.items():
        # Filtrer par quote
        if not symbol.endswith(f"/{quote_filter}"):
            continue

        change_pct = float(data.get("percentage", 0) or 0)
        quote_volume = float(data.get("quoteVolume", 0) or 0)

        # Filtrer par volume
        if quote_volume < min_volume_usdt:
            continue

        # Filtrer les variations extrêmes (probables erreurs)
        if change_pct > 1000 or change_pct < -99:
            continue

        gainers.append({
            "symbol": symbol,
            "change_24h_pct": change_pct,
            "last_price": float(data.get("last", 0)),
            "volume_usdt": quote_volume,
            "high_24h": float(data.get("high", 0)),
            "low_24h": float(data.get("low", 0)),
        })

    # Trier par performance descendante
    gainers.sort(key=lambda x: x["change_24h_pct"], reverse=True)

    return gainers[:limit]

async def get_top_losers(
    self,
    quote_filter: str = "USDT",
    min_volume_usdt: float = 10000,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Top losers (pour détecter les dips potentiels)"""
    try:
        tickers = await self.exchange.fetch_tickers()
    except Exception:
        return []

    losers = []

    for symbol, data in tickers.items():
        if not symbol.endswith(f"/{quote_filter}"):
            continue

        change_pct = float(data.get("percentage", 0) or 0)
        quote_volume = float(data.get("quoteVolume", 0) or 0)

        if quote_volume < min_volume_usdt:
            continue

        if change_pct >= 0 or change_pct < -99:
            continue

        losers.append({
            "symbol": symbol,
            "change_24h_pct": change_pct,
            "last_price": float(data.get("last", 0)),
            "volume_usdt": quote_volume,
        })

    losers.sort(key=lambda x: x["change_24h_pct"])  # Ascending (most negative first)

    return losers[:limit]
```

**Critères de validation**:
- [ ] Filtrage par volume pour qualité
- [ ] Tri correct des résultats
- [ ] Filtrage des anomalies de données
- [ ] Performance acceptable sur gros exchanges

---

### T1.4.3 - Implémenter l'estimation de liquidité
**Priorité**: MOYENNE
**Estimation**: Moyenne

```python
async def estimate_liquidity(
    self,
    symbol: str,
    depth_pct: float = 1.0,  # ±1% du prix
) -> Dict[str, Any]:
    """
    Estime la liquidité disponible autour du prix actuel.
    Utile pour le risk manager avant d'exécuter un gros ordre.
    """
    try:
        ticker = await self.get_ticker(symbol)
        orderbook = await self.get_orderbook(symbol, limit=50)
    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "tradeable": False,
        }

    mid_price = ticker["last"]
    upper_bound = mid_price * (1 + depth_pct / 100)
    lower_bound = mid_price * (1 - depth_pct / 100)

    # Liquidité côté bid (ce qu'on peut vendre)
    bid_liquidity = 0.0
    for price, qty in orderbook["bids"]:
        if price >= lower_bound:
            bid_liquidity += price * qty

    # Liquidité côté ask (ce qu'on peut acheter)
    ask_liquidity = 0.0
    for price, qty in orderbook["asks"]:
        if price <= upper_bound:
            ask_liquidity += price * qty

    return {
        "symbol": symbol,
        "mid_price": mid_price,
        "depth_pct": depth_pct,
        "bid_liquidity_usdt": bid_liquidity,
        "ask_liquidity_usdt": ask_liquidity,
        "total_liquidity_usdt": bid_liquidity + ask_liquidity,
        "spread_pct": ((ticker["ask"] - ticker["bid"]) / mid_price) * 100 if mid_price > 0 else 0,
        "volume_24h_usdt": ticker["quote_volume_24h"],
        "tradeable": bid_liquidity > 100 and ask_liquidity > 100,
    }
```

**Critères de validation**:
- [ ] Calcul précis de la liquidité à ±X%
- [ ] Calcul du spread
- [ ] Flag tradeable pour filtrage rapide
- [ ] Gestion des erreurs pour pairs inexistantes

---

## T1.5 - Mode Paper Trading

### T1.5.1 - Implémenter le simulateur Paper Trading
**Priorité**: CRITIQUE
**Estimation**: Haute

Créer une classe `PaperExchangeClient` qui hérite de `ExchangeClient` et simule les ordres :

```python
class PaperExchangeClient(ExchangeClient):
    """
    Version paper trading de l'exchange client.
    - Les prix sont réels (fetch depuis l'exchange)
    - Les ordres sont simulés en mémoire
    """

    def __init__(self, initial_balance_usdt: float = 1000.0) -> None:
        super().__init__()
        self._paper_balance: Dict[str, float] = {
            get_config().base_currency: initial_balance_usdt
        }
        self._paper_positions: Dict[str, Dict] = {}
        self._paper_orders: List[Dict] = []
        self._order_counter = 0

    async def get_balance(self) -> Dict[str, Any]:
        """Balance simulée"""
        return {
            "total": dict(self._paper_balance),
            "free": dict(self._paper_balance),
            "used": {},
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simule l'exécution d'un ordre"""
        cfg = get_config()
        base, quote = symbol.split("/")

        # Récupérer le prix réel
        real_price = price if price else await self.get_market_price(symbol)
        cost = amount * real_price

        side = side.lower()

        if side == "buy":
            # Vérifier la balance quote
            quote_balance = self._paper_balance.get(quote, 0.0)
            if cost > quote_balance:
                raise ValueError(f"Insufficient {quote} balance: {quote_balance} < {cost}")

            # Exécuter
            self._paper_balance[quote] = quote_balance - cost
            self._paper_balance[base] = self._paper_balance.get(base, 0.0) + amount

            # Tracker position
            if symbol not in self._paper_positions:
                self._paper_positions[symbol] = {
                    "amount": 0.0,
                    "avg_price": 0.0,
                    "cost_basis": 0.0,
                }
            pos = self._paper_positions[symbol]
            old_cost = pos["amount"] * pos["avg_price"]
            new_cost = amount * real_price
            total_amount = pos["amount"] + amount
            if total_amount > 0:
                pos["avg_price"] = (old_cost + new_cost) / total_amount
            pos["amount"] = total_amount
            pos["cost_basis"] = pos["amount"] * pos["avg_price"]

        else:  # sell
            base_balance = self._paper_balance.get(base, 0.0)
            if amount > base_balance:
                amount = base_balance  # Vendre ce qu'on a

            if amount <= 0:
                raise ValueError(f"No {base} to sell")

            revenue = amount * real_price
            self._paper_balance[base] = base_balance - amount
            self._paper_balance[quote] = self._paper_balance.get(quote, 0.0) + revenue

            # Mettre à jour position
            if symbol in self._paper_positions:
                pos = self._paper_positions[symbol]
                pos["amount"] -= amount
                if pos["amount"] <= 0:
                    del self._paper_positions[symbol]
                else:
                    pos["cost_basis"] = pos["amount"] * pos["avg_price"]

        self._order_counter += 1
        order = {
            "id": f"paper_{self._order_counter}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": real_price,
            "cost": cost if side == "buy" else amount * real_price,
            "status": "filled",
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "fee": {"cost": cost * 0.001, "currency": quote},  # 0.1% simulated fee
        }
        self._paper_orders.append(order)

        return order

    def get_paper_pnl(self) -> Dict[str, Any]:
        """Calcule le PnL du paper trading"""
        cfg = get_config()
        initial = 1000.0  # TODO: rendre configurable
        current = self._paper_balance.get(cfg.base_currency, 0.0)

        # Ajouter valorisation des positions
        # (simplifié - en réalité il faudrait fetch les prix actuels)
        for pos in self._paper_positions.values():
            current += pos.get("cost_basis", 0.0)

        return {
            "initial_balance": initial,
            "current_equity": current,
            "pnl_usdt": current - initial,
            "pnl_pct": ((current / initial) - 1) * 100 if initial > 0 else 0,
            "total_trades": len(self._paper_orders),
        }
```

**Critères de validation**:
- [ ] Prix réels, ordres simulés
- [ ] Tracking précis des positions et coûts moyens
- [ ] Simulation des frais (0.1%)
- [ ] Calcul du PnL
- [ ] Pas de modification de l'exchange réel

---

### T1.5.2 - Factory pour choisir le bon client
**Priorité**: HAUTE
**Estimation**: Simple

```python
def create_exchange_client() -> ExchangeClient:
    """
    Factory qui retourne le bon client selon la config.
    - PAPER_TRADING=true -> PaperExchangeClient
    - PAPER_TRADING=false -> ExchangeClient (réel)
    """
    cfg = get_config()

    if cfg.paper_trading:
        initial_balance = float(os.getenv("PAPER_INITIAL_BALANCE", "1000.0"))
        return PaperExchangeClient(initial_balance_usdt=initial_balance)
    else:
        return ExchangeClient()


# Module-level singleton
_client: Optional[ExchangeClient] = None

def get_exchange_client() -> ExchangeClient:
    global _client
    if _client is None:
        _client = create_exchange_client()
    return _client
```

**Critères de validation**:
- [ ] Selection automatique selon config
- [ ] Singleton pour éviter multiples connexions
- [ ] Transparent pour le reste du code

---

## Checklist Finale Phase 1

- [ ] ExchangeClient avec toutes les méthodes de lecture
- [ ] Méthodes d'exécution d'ordres (place_order, close_position)
- [ ] PaperExchangeClient pour simulation
- [ ] Détection nouveaux listings
- [ ] Scan top gainers/losers
- [ ] Estimation de liquidité
- [ ] Factory et singleton
- [ ] Tests unitaires avec mocks CCXT
- [ ] Documentation des méthodes

---

## Notes Techniques

### Rate Limits MEXC
- API publique: 20 req/s
- API privée: 10 req/s
- Utiliser `enableRateLimit=True` dans CCXT

### Spécificités MEXC
- Pas de passphrase (contrairement à OKX)
- Symboles au format `BASE/QUOTE`
- Beaucoup de microcaps avec faible liquidité

### Gestion des Erreurs
```python
from ccxt.base.errors import (
    ExchangeError,
    InsufficientFunds,
    InvalidOrder,
    OrderNotFound,
    RateLimitExceeded,
)
```

Toujours wrapper les appels CCXT avec try/except appropriés.
