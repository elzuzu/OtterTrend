# Phase 1 - Market & Portfolio (MEXC via CCXT)

> **Objectif**: Implémenter les outils OBSERVER et AGIR pour l'interface avec l'exchange MEXC - données marché, portefeuille et exécution d'ordres.
>
> **Référence**: Conversation ChatGPT - Architecture LLM orchestrateur + mini-ML + outils

## Pourquoi MEXC ?

Pour une stratégie **"1% ROI/jour + Trends SocialFi/Memecoins"** avec un **petit capital**, MEXC est le choix optimal :

| Critère | MEXC 🏆 | OKX | Bybit |
|---------|---------|-----|-------|
| **Frais Spot** | **0.00% Maker / 0.01% Taker** | 0.08% / 0.10% | 0.10% / 0.10% |
| **Vitesse Listing** | **Très rapide (Degen)** | Lente | Moyenne |
| **Niches SocialFi/Meme** | **Énorme choix** | Faible | Bon |
| **Liquidité** | Moyenne | Excellent | Excellent |

**Avantages clés pour notre bot :**
1. **Frais quasi nuls** - Critical pour 10-20 trades/jour. Sur OKX, les 0.1% mangent les profits.
2. **Listings agressifs** - Tokens SocialFi disponibles des semaines avant OKX/Binance.
3. **Scalping possible** - Avec 0% fees maker, on peut capturer des mouvements plus petits.

**Note sécurité** : MEXC est une plateforme de **transit et d'exécution**, pas de stockage long terme. Ne pas y laisser de gros montants dormants.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (structure, config, memory)

---

## T1.1 - Interface Exchange MEXC (OBSERVER)

### T1.1.1 - Créer le wrapper CCXT pour MEXC
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `src/tools/market.py` avec support MEXC comme exchange principal.

**Architecture** :
```python
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
from src.config import get_config

class ExchangeClient:
    """
    Interface unifiée pour MEXC via CCXT.
    Outil OBSERVER - données marché brutes.

    MEXC: Frais 0% maker / 0.01% taker - idéal pour high-frequency.
    """

    def __init__(self) -> None:
        self._exchange: Optional[ccxt.Exchange] = None
        self._config = get_config()

    def _create_exchange(self) -> ccxt.Exchange:
        """Factory pour créer l'instance MEXC"""
        params = {
            "apiKey": self._config.mexc_api_key,
            "secret": self._config.mexc_api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "recvWindow": 60000,  # MEXC spécifique
            },
        }

        exchange = ccxt.mexc(params)

        if self._config.mexc_testnet:
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
- [ ] Support MEXC avec recvWindow
- [ ] Mode sandbox configurable
- [ ] Rate limiting activé (MEXC stricte sur les requêtes)
- [ ] Connexion lazy (créée au premier appel)
- [ ] Méthode close() pour cleanup

---

### T1.1.2 - Implémenter get_market_snapshot (OBSERVER Tool)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Outil principal pour le LLM - snapshot complet du marché:

```python
async def get_market_snapshot(
    self,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    OBSERVER TOOL: get_market_snapshot
    Snapshot complet pour le LLM autonome.

    Retourne:
    - timestamp: ISO format
    - tickers: prix, volume, change 24h
    - portfolio: état du portefeuille
    - new_listings: tokens récemment listés (spécialité MEXC)
    """
    if symbols is None:
        # Tokens SocialFi par défaut (ChatGPT spec)
        symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            "CYBER/USDT", "ID/USDT", "DEGEN/USDT",  # SocialFi
        ]

    tickers = await self.get_tickers(symbols)
    portfolio = await self.get_portfolio_state()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": "mexc",
        "mode": "paper" if self._config.paper_trading else "live",
        "fees_info": "0% maker / 0.01% taker - profite pour scalper",
        "tickers": tickers,
        "portfolio": portfolio,
    }
```

**Critères de validation**:
- [ ] Snapshot structuré et complet
- [ ] Timestamp UTC
- [ ] Inclut tokens SocialFi par défaut
- [ ] Performance acceptable (< 5s)

---

### T1.1.3 - Implémenter get_orderbook (OBSERVER Tool)
**Priorité**: HAUTE
**Estimation**: Simple

```python
async def get_orderbook(
    self,
    symbol: str,
    depth: int = 20,
) -> Dict[str, Any]:
    """
    OBSERVER TOOL: get_orderbook
    Carnet d'ordres pour analyse de liquidité.

    Note: Sur MEXC, vérifier la liquidité car certaines paires
    sont moins profondes que sur OKX/Binance.
    """
    ob = await self.exchange.fetch_order_book(symbol, depth)

    # Calcul du spread
    best_bid = float(ob["bids"][0][0]) if ob["bids"] else 0
    best_ask = float(ob["asks"][0][0]) if ob["asks"] else 0
    mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
    spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price else 0

    return {
        "symbol": symbol,
        "timestamp": ob.get("timestamp"),
        "bids": ob["bids"][:depth],
        "asks": ob["asks"][:depth],
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_pct": round(spread_pct, 4),
        # Warning si faible liquidité
        "low_liquidity_warning": spread_pct > 1.0,
    }
```

**Critères de validation**:
- [ ] Profondeur configurable
- [ ] Calcul du spread automatique
- [ ] Warning si liquidité faible
- [ ] Données normalisées

---

### T1.1.4 - Implémenter méthodes de lecture marché
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Ajouter les méthodes helper à `ExchangeClient` :

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

## T1.2 - Gestion du Portefeuille (AGIR - Lecture)

### T1.2.1 - Implémenter get_portfolio_state (AGIR Tool)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

```python
async def get_portfolio_state(self) -> Dict[str, Any]:
    """
    AGIR TOOL: get_portfolio_state
    État complet du portefeuille avec valorisation.

    Retourne:
    - balance_usdt: solde libre en USDT
    - total_equity_usdt: valeur totale du portefeuille
    - positions: liste des positions avec valorisation
    - positions_count: nombre de positions ouvertes
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
        "exchange": "mexc",
        "balance_usdt": base_bal,
        "total_equity_usdt": total_equity,
        "positions": positions,
        "positions_count": len(positions),
    }

async def get_balance(self) -> Dict[str, Any]:
    """Récupère la balance complète du compte"""
    balance = await self.exchange.fetch_balance()
    return {
        "total": {k: float(v) for k, v in balance.get("total", {}).items() if v > 0},
        "free": {k: float(v) for k, v in balance.get("free", {}).items() if v > 0},
        "used": {k: float(v) for k, v in balance.get("used", {}).items() if v > 0},
    }
```

**Critères de validation**:
- [ ] Balance en temps réel
- [ ] Valorisation des positions en USDT
- [ ] Calcul d'equity total
- [ ] Gestion des assets non-tradeables

---

### T1.2.2 - Implémenter risk_constraints (AGIR Tool)
**Priorité**: CRITIQUE
**Estimation**: Simple

```python
def risk_constraints() -> Dict[str, Any]:
    """
    AGIR TOOL: risk_constraints
    Retourne les limites de risque hard-coded (ChatGPT spec).
    Le LLM doit respecter ces contraintes.
    """
    cfg = get_config()
    return {
        "max_order_usd": cfg.max_order_usd,          # $20 max par ordre
        "max_equity_pct": cfg.max_equity_pct,        # 5% max du portefeuille
        "max_daily_loss_usd": cfg.max_daily_loss_usd,  # $50 halt
        "max_positions": cfg.max_positions,          # 5 positions max
        "paper_mode": cfg.paper_trading,
        "exchange_fees": "0% maker / 0.01% taker",   # MEXC advantage
    }
```

**Critères de validation**:
- [ ] Limites provenant de la config
- [ ] Format clair pour le LLM

---

## T1.3 - Exécution d'Ordres (AGIR)

### T1.3.1 - Implémenter place_order (AGIR Tool)
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
    AGIR TOOL: place_order
    Place un ordre sur MEXC.

    Note MEXC: Préférer les ordres LIMIT (0% fees) aux MARKET (0.01% fees)
    quand possible pour maximiser les profits.

    IMPORTANT: La couche risk doit valider AVANT d'appeler cette méthode!
    Le bot autonome appelle directement quand il veut trader.
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
        "fee_info": "0% if limit, 0.01% if market",
    }
```

**Critères de validation**:
- [ ] Support ordres market et limit
- [ ] Retour normalisé avec toutes les infos
- [ ] Gestion des erreurs MEXC
- [ ] Logs de l'exécution

---

### T1.3.2 - Implémenter close_position et cancel_order (AGIR Tools)
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
    """
    AGIR TOOL: close_position
    Ferme toute la position sur un symbole (vend tout).
    """
    base_asset = symbol.split("/")[0]
    balance = await self.get_balance()
    qty = balance["free"].get(base_asset, 0.0)

    if qty <= 0:
        return None

    # Vérifier la taille minimum d'ordre MEXC
    try:
        markets = await self.exchange.load_markets()
        market = markets.get(symbol)
        if market:
            min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
            if qty < min_amount:
                return {"error": f"Quantity {qty} below minimum {min_amount}"}
    except Exception:
        pass

    return await self.place_order(symbol, "sell", qty, "market")

async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
    """
    AGIR TOOL: cancel_order
    Annule un ordre ouvert.
    """
    try:
        result = await self.exchange.cancel_order(order_id, symbol)
        return {"success": True, "order_id": order_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
```

**Critères de validation**:
- [ ] close_position vend la totalité
- [ ] cancel_order avec gestion d'erreurs
- [ ] Retours normalisés

---

## T1.4 - Intelligence Market (OBSERVER Avancé)

### T1.4.1 - Implémenter get_trending_tokens (OBSERVER Tool)
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def get_trending_tokens(
    self,
    quote_filter: str = "USDT",
    min_volume_usdt: float = 50000,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    OBSERVER TOOL: get_trending_tokens
    Top gainers et losers pour détecter les mouvements.

    MEXC: Exchange spécialisé dans les listings rapides.
    Les nouveaux tokens apparaissent ici avant OKX/Binance.
    """
    try:
        tickers = await self.exchange.fetch_tickers()
    except Exception as e:
        return {"error": str(e), "gainers": [], "losers": []}

    filtered = []

    for symbol, data in tickers.items():
        if not symbol.endswith(f"/{quote_filter}"):
            continue

        change_pct = float(data.get("percentage", 0) or 0)
        quote_volume = float(data.get("quoteVolume", 0) or 0)

        if quote_volume < min_volume_usdt:
            continue

        # Filtrer les anomalies
        if change_pct > 500 or change_pct < -95:
            continue

        filtered.append({
            "symbol": symbol,
            "change_24h_pct": round(change_pct, 2),
            "last_price": float(data.get("last", 0)),
            "volume_usdt": round(quote_volume, 0),
        })

    # Trier pour gainers et losers
    sorted_list = sorted(filtered, key=lambda x: x["change_24h_pct"], reverse=True)

    return {
        "gainers": sorted_list[:limit],
        "losers": sorted_list[-limit:][::-1],
        "total_symbols": len(filtered),
        "exchange_note": "MEXC liste agressivement les nouveaux tokens - surveiller les récents listings",
    }
```

**Critères de validation**:
- [ ] Filtrage par volume pour qualité
- [ ] Tri correct des résultats
- [ ] Filtrage des anomalies

---

### T1.4.2 - Implémenter detect_new_listings (OBSERVER Tool - MEXC Spécialité)
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
async def detect_new_listings(
    self,
    known_symbols: Optional[set] = None,
    quote_filter: str = "USDT",
) -> List[Dict[str, Any]]:
    """
    OBSERVER TOOL: detect_new_listings
    Détecte les nouveaux listings - LA spécialité de MEXC!

    MEXC liste souvent des tokens des semaines avant OKX/Binance.
    C'est un avantage stratégique majeur pour notre bot.
    """
    markets = await self.exchange.load_markets(reload=True)

    current_symbols = {
        symbol for symbol, market in markets.items()
        if market.get("quote") == quote_filter and market.get("active")
    }

    if known_symbols is None:
        # Premier appel - retourner les symboles actuels
        return {"symbols": list(current_symbols), "new_count": 0}

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
                "tip": "Nouveau listing MEXC - potentiel early entry",
            })
        except Exception:
            continue

    return {
        "new_listings": new_listings,
        "new_count": len(new_listings),
        "strategy_note": "Les nouveaux listings MEXC précèdent souvent OKX/Binance de plusieurs semaines",
    }
```

**Critères de validation**:
- [ ] Détection efficace des nouveaux tokens
- [ ] Enrichissement avec ticker data
- [ ] Note stratégique pour le LLM

---

### T1.4.3 - Implémenter ml_estimate_slippage (RÉFLÉCHIR Tool)
**Priorité**: MOYENNE
**Estimation**: Moyenne

```python
async def ml_estimate_slippage(
    self,
    symbol: str,
    side: str,  # 'buy' | 'sell'
    amount_usdt: float,
) -> Dict[str, Any]:
    """
    RÉFLÉCHIR TOOL: ml_estimate_slippage
    Estime le slippage pour un ordre donné basé sur l'orderbook.

    Important sur MEXC: Liquidité parfois faible sur les microcaps.
    Vérifier avant d'exécuter des ordres significatifs.
    """
    try:
        ob = await self.get_orderbook(symbol, depth=50)
        ticker = await self.get_ticker(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "tradeable": False}

    mid_price = ob["mid_price"]

    # Calculer le slippage estimé
    if side.lower() == "buy":
        book = ob["asks"]
    else:
        book = ob["bids"]

    remaining_usdt = amount_usdt
    weighted_price = 0.0
    total_filled = 0.0

    for price, qty in book:
        level_value = price * qty
        if level_value >= remaining_usdt:
            fill_qty = remaining_usdt / price
            weighted_price += price * fill_qty
            total_filled += fill_qty
            remaining_usdt = 0
            break
        else:
            weighted_price += price * qty
            total_filled += qty
            remaining_usdt -= level_value

    if total_filled > 0:
        avg_fill_price = weighted_price / total_filled
        slippage_pct = ((avg_fill_price - mid_price) / mid_price) * 100
        if side.lower() == "sell":
            slippage_pct = -slippage_pct
    else:
        avg_fill_price = mid_price
        slippage_pct = 0

    # MEXC: plus strict sur la liquidité acceptable
    tradeable = ob["spread_pct"] < 1.5 and abs(slippage_pct) < 0.5

    return {
        "symbol": symbol,
        "side": side,
        "amount_usdt": amount_usdt,
        "mid_price": mid_price,
        "estimated_fill_price": round(avg_fill_price, 8),
        "slippage_pct": round(abs(slippage_pct), 4),
        "spread_pct": ob["spread_pct"],
        "tradeable": tradeable,
        "mexc_note": "Attention aux microcaps avec faible liquidité" if not tradeable else "Liquidité OK",
    }
```

**Critères de validation**:
- [ ] Calcul précis du slippage
- [ ] Flag tradeable adapté à MEXC
- [ ] Gestion des erreurs

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
    - Les prix sont réels (fetch depuis MEXC)
    - Les ordres sont simulés en mémoire
    - Frais simulés: 0% maker / 0.01% taker (avantage MEXC!)
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
        """Simule l'exécution d'un ordre avec prix réels MEXC"""
        cfg = get_config()
        base, quote = symbol.split("/")

        # Récupérer le prix réel depuis MEXC
        real_price = price if price else await self.get_market_price(symbol)
        cost = amount * real_price

        # MEXC fees: 0% maker, 0.01% taker
        fee_rate = 0.0001 if order_type == "market" else 0.0  # 0.01% ou 0%

        side = side.lower()

        if side == "buy":
            fee = cost * fee_rate
            total_cost = cost + fee
            quote_balance = self._paper_balance.get(quote, 0.0)

            if total_cost > quote_balance:
                raise ValueError(f"Insufficient {quote} balance: {quote_balance:.2f} < {total_cost:.2f}")

            self._paper_balance[quote] = quote_balance - total_cost
            self._paper_balance[base] = self._paper_balance.get(base, 0.0) + amount

        else:  # sell
            base_balance = self._paper_balance.get(base, 0.0)
            if amount > base_balance:
                amount = base_balance

            if amount <= 0:
                raise ValueError(f"No {base} to sell")

            revenue = amount * real_price
            fee = revenue * fee_rate
            net_revenue = revenue - fee

            self._paper_balance[base] = base_balance - amount
            self._paper_balance[quote] = self._paper_balance.get(quote, 0.0) + net_revenue

        self._order_counter += 1
        order = {
            "id": f"paper_{self._order_counter}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": real_price,
            "cost": cost,
            "status": "filled",
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "fee": {"cost": fee, "currency": quote, "rate": fee_rate},
        }
        self._paper_orders.append(order)

        return order

    def get_paper_pnl(self) -> Dict[str, Any]:
        """Calcule le PnL du paper trading"""
        cfg = get_config()
        initial = 1000.0
        current = self._paper_balance.get(cfg.base_currency, 0.0)

        return {
            "initial_balance": initial,
            "current_usdt": current,
            "pnl_usdt": current - initial,
            "pnl_pct": ((current / initial) - 1) * 100 if initial > 0 else 0,
            "total_trades": len(self._paper_orders),
            "fees_advantage": "MEXC 0%/0.01% vs OKX 0.08%/0.10% - économies significatives",
        }
```

**Critères de validation**:
- [ ] Prix réels MEXC, ordres simulés
- [ ] Simulation des frais MEXC (0% maker / 0.01% taker)
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
    - PAPER_TRADING=false -> ExchangeClient (réel MEXC)
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

### Outils OBSERVER
- [ ] get_market_snapshot - snapshot complet marché
- [ ] get_orderbook - carnet d'ordres avec spread
- [ ] get_ticker/get_tickers - prix temps réel
- [ ] get_ohlcv - chandeliers historiques
- [ ] get_trending_tokens - gainers/losers
- [ ] detect_new_listings - nouveaux tokens (spécialité MEXC)

### Outils AGIR (Portfolio)
- [ ] get_portfolio_state - état portefeuille valorisé
- [ ] get_balance - balance brute
- [ ] risk_constraints - limites de risque

### Outils AGIR (Exécution)
- [ ] place_order - passer des ordres
- [ ] close_position - fermer une position
- [ ] cancel_order - annuler un ordre

### Outils RÉFLÉCHIR
- [ ] ml_estimate_slippage - estimation slippage

### Infrastructure
- [ ] PaperExchangeClient pour simulation
- [ ] Factory et singleton
- [ ] Tests unitaires avec mocks CCXT

---

## Notes Techniques MEXC

### Rate Limits MEXC
- API publique: 20 req/s
- API privée: 10 req/s
- **Plus stricte que OKX** - utiliser `enableRateLimit=True` obligatoirement
- Ajouter un délai de 100ms entre les appels si nécessaire

### Spécificités MEXC
- **Pas de passphrase** (contrairement à OKX) - juste API key + secret
- **recvWindow**: Requis, typiquement 60000ms
- Symboles au format `BASE/QUOTE`
- **Beaucoup de microcaps** - attention à la liquidité

### Avantages MEXC pour notre stratégie
- **Frais 0% maker** - idéal pour ordres limite
- **Listings rapides** - tokens dispo avant OKX/Binance
- **Scalping possible** - petits mouvements rentables grâce aux frais bas

### Gestion des Erreurs MEXC
```python
from ccxt.base.errors import (
    ExchangeError,
    InsufficientFunds,
    InvalidOrder,
    OrderNotFound,
    RateLimitExceeded,
    RequestTimeout,  # Plus fréquent sur MEXC
)
```

### Instruction System Prompt (à ajouter)
> "Tu trades sur MEXC. Profite des frais extrêmement bas (0% maker) pour capturer des mouvements de prix plus petits (scalping) si la tendance est incertaine. Surveille les nouveaux listings récents car c'est la spécialité de cet exchange."

---

## T1.6 - Architecture Modulaire Exchange

> **Objectif**: Assurer que l'implémentation MEXC respecte l'interface `BaseExchange`.
> Permet de swapper facilement MEXC ↔ Binance ↔ OKX ↔ Paper Trading.

### T1.6.1 - Implémenter MEXCExchange (BaseExchange)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Modifier `src/tools/market.py` pour implémenter l'interface :

```python
"""
Implémentation MEXC de l'interface BaseExchange.
Modulaire et swappable avec d'autres exchanges.
"""

from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt

from src.interfaces import BaseExchange
from src.config import get_config


class MEXCExchange(BaseExchange):
    """
    Implémentation concrète de BaseExchange pour MEXC.

    Caractéristiques MEXC:
    - Frais: 0% maker / 0.01% taker
    - Pas de passphrase (contrairement à OKX)
    - Listings très rapides (tokens SocialFi)
    - Rate limits plus stricts
    """

    def __init__(self) -> None:
        self._exchange: Optional[ccxt.Exchange] = None
        self._config = get_config()

    @property
    def name(self) -> str:
        return "mexc"

    @property
    def fees(self) -> Dict[str, float]:
        return {"maker": 0.0, "taker": 0.0001}  # 0% / 0.01%

    def _get_exchange(self) -> ccxt.Exchange:
        """Lazy initialization de l'exchange CCXT"""
        if self._exchange is None:
            self._exchange = ccxt.mexc({
                "apiKey": self._config.mexc_api_key,
                "secret": self._config.mexc_api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "recvWindow": 60000,
                },
            })
        return self._exchange

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Implémente BaseExchange.get_ticker"""
        ticker = await self._get_exchange().fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": float(ticker["last"]),
            "bid": float(ticker.get("bid", 0)),
            "ask": float(ticker.get("ask", 0)),
            "volume_24h": float(ticker.get("baseVolume", 0)),
            "change_24h_pct": float(ticker.get("percentage", 0)),
        }

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """Implémente BaseExchange.get_tickers"""
        tickers = await self._get_exchange().fetch_tickers(symbols)
        return {
            sym: {
                "last": float(data["last"]),
                "volume_24h": float(data.get("baseVolume", 0)),
                "change_24h_pct": float(data.get("percentage", 0)),
            }
            for sym, data in tickers.items()
        }

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Implémente BaseExchange.get_orderbook"""
        ob = await self._get_exchange().fetch_order_book(symbol, depth)
        best_bid = float(ob["bids"][0][0]) if ob["bids"] else 0
        best_ask = float(ob["asks"][0][0]) if ob["asks"] else 0
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
        spread_pct = ((best_ask - best_bid) / mid * 100) if mid else 0

        return {
            "symbol": symbol,
            "bids": ob["bids"][:depth],
            "asks": ob["asks"][:depth],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_pct": round(spread_pct, 4),
        }

    async def get_balance(self) -> Dict[str, Any]:
        """Implémente BaseExchange.get_balance"""
        balance = await self._get_exchange().fetch_balance()
        return {
            "total": {k: float(v) for k, v in balance["total"].items() if v > 0},
            "free": {k: float(v) for k, v in balance["free"].items() if v > 0},
            "used": {k: float(v) for k, v in balance["used"].items() if v > 0},
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Implémente BaseExchange.place_order"""
        exchange = self._get_exchange()

        if order_type == "market":
            order = await exchange.create_market_order(symbol, side, amount)
        elif order_type == "limit":
            if price is None:
                raise ValueError("Prix requis pour ordre limit")
            order = await exchange.create_limit_order(symbol, side, amount, price)
        else:
            raise ValueError(f"Type d'ordre non supporté: {order_type}")

        return {
            "id": order["id"],
            "symbol": order["symbol"],
            "side": order["side"],
            "type": order["type"],
            "amount": float(order["amount"]),
            "price": float(order.get("average") or order.get("price") or 0),
            "status": order["status"],
            "fee": order.get("fee"),
        }

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Implémente BaseExchange.cancel_order"""
        try:
            await self._get_exchange().cancel_order(order_id, symbol)
            return {"success": True, "order_id": order_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close(self) -> None:
        """Implémente BaseExchange.close"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
```

**Critères de validation**:
- [ ] Implémente toutes les méthodes de BaseExchange
- [ ] Propriétés name et fees correctes
- [ ] Tests avec mock CCXT
- [ ] Documentation des spécificités MEXC

---

### T1.6.2 - Implémenter PaperExchange (BaseExchange)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

```python
class PaperExchange(BaseExchange):
    """
    Implémentation paper trading de BaseExchange.
    Prix réels via MEXC, ordres simulés en mémoire.

    Permet de tester le bot sans risque réel.
    Simule les frais MEXC (0% maker / 0.01% taker).
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        real_exchange: Optional[BaseExchange] = None,
    ) -> None:
        self._real_exchange = real_exchange or MEXCExchange()
        self._balance: Dict[str, float] = {"USDT": initial_balance}
        self._orders: List[Dict] = []
        self._order_counter = 0

    @property
    def name(self) -> str:
        return "paper"

    @property
    def fees(self) -> Dict[str, float]:
        # Simule les frais MEXC
        return {"maker": 0.0, "taker": 0.0001}

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Prix réels depuis l'exchange sous-jacent"""
        return await self._real_exchange.get_ticker(symbol)

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """Prix réels depuis l'exchange sous-jacent"""
        return await self._real_exchange.get_tickers(symbols)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Orderbook réel depuis l'exchange sous-jacent"""
        return await self._real_exchange.get_orderbook(symbol, depth)

    async def get_balance(self) -> Dict[str, Any]:
        """Balance simulée"""
        return {
            "total": dict(self._balance),
            "free": dict(self._balance),
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
        """Simule un ordre avec prix réels"""
        base, quote = symbol.split("/")

        # Prix réel
        if price is None:
            ticker = await self.get_ticker(symbol)
            price = ticker["last"]

        cost = amount * price
        fee_rate = self.fees["taker"] if order_type == "market" else self.fees["maker"]

        if side.lower() == "buy":
            fee = cost * fee_rate
            total_cost = cost + fee
            if total_cost > self._balance.get(quote, 0):
                raise ValueError(f"Insufficient {quote}")
            self._balance[quote] = self._balance.get(quote, 0) - total_cost
            self._balance[base] = self._balance.get(base, 0) + amount
        else:
            if amount > self._balance.get(base, 0):
                raise ValueError(f"Insufficient {base}")
            revenue = cost * (1 - fee_rate)
            self._balance[base] = self._balance.get(base, 0) - amount
            self._balance[quote] = self._balance.get(quote, 0) + revenue

        self._order_counter += 1
        order = {
            "id": f"paper_{self._order_counter}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price,
            "status": "filled",
            "fee": {"rate": fee_rate},
        }
        self._orders.append(order)
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Les ordres paper sont toujours remplis immédiatement"""
        return {"success": False, "error": "Paper orders are filled immediately"}

    async def close(self) -> None:
        """Ferme l'exchange sous-jacent"""
        await self._real_exchange.close()

    def get_pnl(self) -> Dict[str, Any]:
        """Méthode additionnelle pour le paper trading"""
        return {
            "total_trades": len(self._orders),
            "balance": self._balance,
        }
```

**Critères de validation**:
- [ ] Même interface que MEXCExchange
- [ ] Prix réels, ordres simulés
- [ ] Simulation correcte des frais
- [ ] Tests unitaires sans API

---

### T1.6.3 - Factory pour exchanges (Pattern Strategy)
**Priorité**: HAUTE
**Estimation**: Simple

```python
"""
Factory pour créer le bon exchange selon la config.
Pattern Strategy pour swapper les implémentations.
"""

from typing import Optional
from src.interfaces import BaseExchange
from src.config import get_config


# Registre des exchanges disponibles
EXCHANGE_REGISTRY: Dict[str, type] = {}


def register_exchange(name: str):
    """Décorateur pour enregistrer un exchange"""
    def decorator(cls):
        EXCHANGE_REGISTRY[name] = cls
        return cls
    return decorator


@register_exchange("mexc")
class MEXCExchange(BaseExchange):
    ...  # Voir implémentation ci-dessus


@register_exchange("paper")
class PaperExchange(BaseExchange):
    ...  # Voir implémentation ci-dessus


# Placeholder pour futures implémentations
@register_exchange("binance")
class BinanceExchange(BaseExchange):
    """TODO: Implémenter si besoin de migrer"""
    ...


def create_exchange(
    exchange_name: Optional[str] = None,
    **kwargs,
) -> BaseExchange:
    """
    Factory pour créer l'exchange approprié.

    Args:
        exchange_name: Nom de l'exchange (mexc, paper, binance)
        **kwargs: Arguments passés au constructeur

    Usage:
        exchange = create_exchange("mexc")
        exchange = create_exchange("paper", initial_balance=500)
    """
    cfg = get_config()

    # Déterminer l'exchange à utiliser
    if exchange_name is None:
        exchange_name = "paper" if cfg.paper_trading else "mexc"

    if exchange_name not in EXCHANGE_REGISTRY:
        available = list(EXCHANGE_REGISTRY.keys())
        raise ValueError(f"Exchange '{exchange_name}' non supporté. Disponibles: {available}")

    exchange_cls = EXCHANGE_REGISTRY[exchange_name]
    return exchange_cls(**kwargs)
```

**Critères de validation**:
- [ ] Pattern Strategy fonctionnel
- [ ] Registre extensible
- [ ] Factory avec détection automatique
- [ ] Placeholders pour futures implémentations

---

## T1.7 - Stack réseau « Envoy-like » pour CEX (état 12/2025)

**Objectif**: intégrer nativement les patterns Envoy Proxy 2025 (circuit breaker, hedging, outlier detection, adaptive concurrency) dans la couche réseau Python pour optimiser les appels REST/WebSocket aux CEX et conserver un avantage net sur les autres bots.

### T1.7.1 - Pooling HTTP/2 + keep-alive agressif
**Priorité**: CRITIQUE  \
**Estimation**: Moyenne

- Client `ResilientHttpClient` basé sur `httpx.AsyncClient` (`http2=True`, TLS reuse, `trust_env=False`) avec pool préchauffé (2-3 connexions/host), limites `max_keepalive_connections`/`max_connections` configurables et timeouts granulaire <3s.
- Keep-alive 60-120s, DNS caching 60s, happy-eyeballs (v4/v6) + fallback IPv4 si p95 latence dépasse un seuil ; compression gzip/brotli auto.
- Hook pour passer ce client dans CCXT (`session=httpx_client`) ou wrapper `aiohttp` avec réglages équivalents.

**Critères de validation**:
- [ ] HTTP/2 actif et testé (dry-run ou ping API).
- [ ] Pool préchauffé observable (metrics connexions, taux de réutilisation).
- [ ] Timeouts/keep-alive paramétrables via `config`.

---

### T1.7.2 - Circuit breaker + retries + hedging
**Priorité**: CRITIQUE  \
**Estimation**: Moyenne

- Circuit breaker par route (tickers, orderbook, orders) avec fenêtres 50-100 requêtes; seuil d'échec >20% ou p95>800ms → open 30-60s; demi-ouverture 1-3 probes.
- Retries idempotents (GET/fetch) max 3 avec backoff exponentiel + jitter (100→800ms) en respectant les budgets rate limit.
- Request hedging (Envoy 2025): lancer une requête miroir quand la première dépasse un deadline, annuler la plus lente.
- Outlier detection: score adaptatif pour évincer temporairement endpoints/IP lents ou en 429/5xx; export de l'état breaker/outlier dans les logs.
- Erreurs typées (enum) propagées au risk manager pour décider d'un ralentissement ou d'un halt partiel.

**Critères de validation**:
- [ ] Breaker configurable par route + états (closed/open/half-open) visibles dans les traces.
- [ ] Hedging activable sur `get_tickers`/`fetch_order_book`.
- [ ] Retries avec jitter et budget respecté.

---

### T1.7.3 - Rate limiting & QoS adaptatif
**Priorité**: HAUTE  \
**Estimation**: Moyenne

- Limiteur token-bucket `aiolimiter` multi-niveaux: global, par host, par type (public/private), par symbole ; quotas configurables.
- File de priorité (orders > account > market bulk) avec temps d'attente borné; load shedding explicite quand la file dépasse le seuil.
- Adaptive concurrency (pattern Envoy adaptive concurrency 2025): ajuster dynamiquement le nombre de requêtes concurrentes selon la latence glissante et le taux d'erreur.
- Cooldown automatique après shed pour éviter l'emballement; hooks pour remonter l'état au LLM.

**Critères de validation**:
- [ ] Budgets rate limit ajustables via `config`.
- [ ] Priorisation vérifiée via tests concurrents (orders passent en premier).
- [ ] Concurrency qui s'adapte à la latence observée.

---

### T1.7.4 - Health-checks actifs + failover multi-endpoints
**Priorité**: HAUTE  \
**Estimation**: Moyenne

- Sondes actives REST (ping/time) et WebSocket (subscribe minimal) toutes les 10-30s avec seuils configurables.
- Bascule automatique vers endpoint/region alternatif (ou PaperExchange) en cas d'état « unhealthy » avec cooldown et backoff; blacklist temporaire des IP en erreur; rotation sur DNS multi-A/AAAA.
- Exposer l'état santé par endpoint dans `get_market_snapshot` et dans les logs.

**Critères de validation**:
- [ ] Health state visible par le LLM/outils OBSERVER.
- [ ] Failover validé via scénarios de panne simulés.

---

### T1.7.5 - Observabilité OpenTelemetry (latence/saturation)
**Priorité**: HAUTE  \
**Estimation**: Moyenne

- Tracing OpenTelemetry autour des calls CCXT/httpx: attributs endpoint, symbol, status, latence, retries, breaker/hedging state, shed/load.
- Metrics Prometheus: taux succès/échec, p50/p95 latence, connexions actives, shed rate, pending queue length, breaker opens.
- Logs JSON corrélés (trace_id/span_id, action LLM) et événements réseau (open/close breaker, failover, shed) ; export OTLP toggle.
- Dashboard minimal (Grafana/console) décrivant la santé réseau + alertes locales (seuil latence, erreurs 429/5xx, shed).

**Critères de validation**:
- [ ] Traces exportables (OTLP localhost) avec activation via `config`.
- [ ] Metrics nommées (`cex_http_latency_ms`, `cex_shed_total`, `cex_breaker_open_total`).
- [ ] Logs corrélés aux décisions du bot.
