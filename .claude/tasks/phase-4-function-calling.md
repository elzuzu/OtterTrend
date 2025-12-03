# Phase 4 - LLM Function Calling

> **Objectif**: Implémenter le système de function calling pour que le LLM puisse appeler directement les outils (market, trends, execution) de manière autonome.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (GroqAdapter)
- Phase 1 complète (market tools)
- Phase 2 complète (trends tools)
- Phase 3 partielle (renderer pour affichage)

**Note MVP** : le produit actuel tourne **sans function calling** (LLM renvoie un JSON `actions`). Cette phase est prévue V2 : ne pas impacter le MVP minimal tant que les fondations précédentes ne sont pas finalisées.

---

## T4.1 - Définition des Schemas de Tools

### T4.1.1 - Créer src/tools/schemas.py avec tous les tools
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Définir les schemas JSON pour tous les outils disponibles au LLM :

```python
"""
Schemas JSON des tools pour le function calling Groq/OpenAI.
Format: OpenAI function calling schema
"""

from typing import List, Dict, Any

# === MARKET TOOLS ===

TOOL_GET_MARKET_PRICE = {
    "type": "function",
    "function": {
        "name": "get_market_price",
        "description": "Get the current spot price for a trading pair",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')",
                },
            },
            "required": ["symbol"],
        },
    },
}

TOOL_GET_TICKER = {
    "type": "function",
    "function": {
        "name": "get_ticker",
        "description": "Get detailed ticker info including price, volume, and 24h change",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol",
                },
            },
            "required": ["symbol"],
        },
    },
}

TOOL_GET_PORTFOLIO = {
    "type": "function",
    "function": {
        "name": "get_portfolio_state",
        "description": "Get current portfolio state including balance, equity, and open positions",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOL_GET_TOP_GAINERS = {
    "type": "function",
    "function": {
        "name": "get_top_gainers",
        "description": "Get top gaining tokens in the last 24h, filtered by volume",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
                "min_volume_usdt": {
                    "type": "number",
                    "description": "Minimum 24h volume in USDT (default: 10000)",
                    "default": 10000,
                },
            },
            "required": [],
        },
    },
}

TOOL_GET_NEW_LISTINGS = {
    "type": "function",
    "function": {
        "name": "get_new_listings",
        "description": "Get recently listed tokens on the exchange",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOL_ESTIMATE_LIQUIDITY = {
    "type": "function",
    "function": {
        "name": "estimate_liquidity",
        "description": "Estimate available liquidity around the current price for a symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol",
                },
                "depth_pct": {
                    "type": "number",
                    "description": "Price depth percentage (default: 1.0 for ±1%)",
                    "default": 1.0,
                },
            },
            "required": ["symbol"],
        },
    },
}

# === TRENDS TOOLS ===

TOOL_GET_GOOGLE_TRENDS = {
    "type": "function",
    "function": {
        "name": "get_google_trends",
        "description": "Get Google Trends interest scores for keywords",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords to analyze (max 5)",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Time range ('now 1-H', 'now 7-d', 'today 1-m')",
                    "default": "now 7-d",
                },
            },
            "required": ["keywords"],
        },
    },
}

TOOL_GET_NEWS_SENTIMENT = {
    "type": "function",
    "function": {
        "name": "get_news_sentiment",
        "description": "Get news sentiment analysis for a crypto symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol to analyze",
                },
            },
            "required": ["symbol"],
        },
    },
}

TOOL_GET_TREND_SNAPSHOT = {
    "type": "function",
    "function": {
        "name": "get_trend_snapshot",
        "description": "Get a complete snapshot of current trends (Google Trends + news sentiment)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# === EXECUTION TOOLS ===

TOOL_PLACE_ORDER = {
    "type": "function",
    "function": {
        "name": "place_order",
        "description": "Place a market order. Order will be validated by risk manager before execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol (e.g., 'BTC/USDT')",
                },
                "side": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Order side",
                },
                "amount_usd": {
                    "type": "number",
                    "description": "Order amount in USD (will be converted to token amount)",
                },
            },
            "required": ["symbol", "side", "amount_usd"],
        },
    },
}

TOOL_CLOSE_POSITION = {
    "type": "function",
    "function": {
        "name": "close_position",
        "description": "Close entire position for a symbol (sell all holdings)",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol to close",
                },
            },
            "required": ["symbol"],
        },
    },
}

# === ANALYTICS TOOLS ===

TOOL_GET_MARKET_ANALYTICS = {
    "type": "function",
    "function": {
        "name": "get_market_analytics",
        "description": "Get technical analytics for a symbol (regime, volatility, RSI)",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol",
                },
            },
            "required": ["symbol"],
        },
    },
}

# === ALL TOOLS ===

ALL_TOOLS: List[Dict[str, Any]] = [
    # Market (read)
    TOOL_GET_MARKET_PRICE,
    TOOL_GET_TICKER,
    TOOL_GET_PORTFOLIO,
    TOOL_GET_TOP_GAINERS,
    TOOL_GET_NEW_LISTINGS,
    TOOL_ESTIMATE_LIQUIDITY,
    # Trends (read)
    TOOL_GET_GOOGLE_TRENDS,
    TOOL_GET_NEWS_SENTIMENT,
    TOOL_GET_TREND_SNAPSHOT,
    # Execution (write)
    TOOL_PLACE_ORDER,
    TOOL_CLOSE_POSITION,
    # Analytics (read)
    TOOL_GET_MARKET_ANALYTICS,
]

# Tools sûrs (lecture seule)
SAFE_TOOLS = [t for t in ALL_TOOLS if t["function"]["name"] not in ("place_order", "close_position")]

# Tools d'exécution (nécessitent validation)
EXECUTION_TOOLS = [t for t in ALL_TOOLS if t["function"]["name"] in ("place_order", "close_position")]


def get_tool_names() -> List[str]:
    """Retourne la liste des noms de tools"""
    return [t["function"]["name"] for t in ALL_TOOLS]
```

**Critères de validation**:
- [ ] Tous les tools définis avec descriptions claires
- [ ] Types corrects (string, number, array, etc.)
- [ ] Required vs optional bien définis
- [ ] Séparation tools lecture vs exécution

---

## T4.2 - Router de Tools

### T4.2.1 - Créer le router qui exécute les tools
**Priorité**: CRITIQUE
**Estimation**: Haute

Créer `src/tools/router.py` :

```python
"""
Router pour exécuter les tools appelés par le LLM.
Mappe les noms de functions aux implémentations réelles.
"""

import asyncio
from typing import Any, Callable, Dict, Optional
from functools import wraps

from src.tools import market, trends, analytics
from src.tools.risk import RiskManager
from src.config import get_config


class ToolExecutionError(Exception):
    """Erreur lors de l'exécution d'un tool"""
    pass


class ToolRouter:
    """
    Route les appels de tools du LLM vers les implémentations.
    Gère la validation, l'exécution et le formatting des résultats.
    """

    def __init__(self, risk_manager: Optional[RiskManager] = None) -> None:
        self.risk_manager = risk_manager or RiskManager()
        self._exchange_client = None
        self._trend_analyzer = None
        self._handlers: Dict[str, Callable] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Enregistre tous les handlers de tools"""
        # Market tools
        self._handlers["get_market_price"] = self._handle_get_market_price
        self._handlers["get_ticker"] = self._handle_get_ticker
        self._handlers["get_portfolio_state"] = self._handle_get_portfolio
        self._handlers["get_top_gainers"] = self._handle_get_top_gainers
        self._handlers["get_new_listings"] = self._handle_get_new_listings
        self._handlers["estimate_liquidity"] = self._handle_estimate_liquidity

        # Trends tools
        self._handlers["get_google_trends"] = self._handle_get_google_trends
        self._handlers["get_news_sentiment"] = self._handle_get_news_sentiment
        self._handlers["get_trend_snapshot"] = self._handle_get_trend_snapshot

        # Execution tools
        self._handlers["place_order"] = self._handle_place_order
        self._handlers["close_position"] = self._handle_close_position

        # Analytics tools
        self._handlers["get_market_analytics"] = self._handle_get_market_analytics

    @property
    def exchange_client(self):
        """Lazy loading de l'exchange client"""
        if self._exchange_client is None:
            from src.tools.market import get_exchange_client
            self._exchange_client = get_exchange_client()
        return self._exchange_client

    @property
    def trend_analyzer(self):
        """Lazy loading du trend analyzer"""
        if self._trend_analyzer is None:
            from src.tools.trends import get_trend_analyzer
            self._trend_analyzer = get_trend_analyzer()
        return self._trend_analyzer

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Exécute un tool et retourne le résultat.

        Args:
            tool_name: Nom du tool à exécuter
            arguments: Arguments passés par le LLM

        Returns:
            Dict avec le résultat ou l'erreur
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            result = await handler(arguments)
            return {
                "success": True,
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # === MARKET HANDLERS ===

    async def _handle_get_market_price(self, args: Dict) -> float:
        symbol = args["symbol"]
        return await self.exchange_client.get_market_price(symbol)

    async def _handle_get_ticker(self, args: Dict) -> Dict:
        symbol = args["symbol"]
        return await self.exchange_client.get_ticker(symbol)

    async def _handle_get_portfolio(self, args: Dict) -> Dict:
        return await self.exchange_client.get_portfolio_state()

    async def _handle_get_top_gainers(self, args: Dict) -> list:
        limit = args.get("limit", 10)
        min_vol = args.get("min_volume_usdt", 10000)
        return await self.exchange_client.get_top_gainers(
            limit=limit, min_volume_usdt=min_vol
        )

    async def _handle_get_new_listings(self, args: Dict) -> list:
        return await self.exchange_client.detect_new_listings()

    async def _handle_estimate_liquidity(self, args: Dict) -> Dict:
        symbol = args["symbol"]
        depth = args.get("depth_pct", 1.0)
        return await self.exchange_client.estimate_liquidity(symbol, depth)

    # === TRENDS HANDLERS ===

    async def _handle_get_google_trends(self, args: Dict) -> Dict:
        keywords = args["keywords"][:5]  # Limit to 5
        timeframe = args.get("timeframe", "now 7-d")
        return await self.trend_analyzer.trends_client.get_interest_over_time(
            keywords, timeframe
        )

    async def _handle_get_news_sentiment(self, args: Dict) -> Dict:
        symbol = args["symbol"]
        news = await self.trend_analyzer.news_client.search_news_by_symbol(symbol)
        from src.tools.trends import analyze_news_sentiment
        return await analyze_news_sentiment(news)

    async def _handle_get_trend_snapshot(self, args: Dict) -> Dict:
        return await self.trend_analyzer.get_trend_snapshot()

    # === EXECUTION HANDLERS ===

    async def _handle_place_order(self, args: Dict) -> Dict:
        symbol = args["symbol"]
        side = args["side"]
        amount_usd = args["amount_usd"]

        # Get current portfolio for risk check
        portfolio = await self.exchange_client.get_portfolio_state()
        balance = portfolio.get("balance_usdt", 0)

        # Risk validation
        ok, reason = self.risk_manager.check_order(
            amount_usd=amount_usd,
            balance_usd=balance,
            symbol=symbol,
        )
        if not ok:
            return {
                "executed": False,
                "reason": f"Risk check failed: {reason}",
            }

        # Liquidity check
        liquidity = await self.exchange_client.estimate_liquidity(symbol)
        if not liquidity.get("tradeable", False):
            return {
                "executed": False,
                "reason": "Insufficient liquidity",
            }

        # Convert USD to token amount
        price = await self.exchange_client.get_market_price(symbol)
        amount = amount_usd / price

        # Execute
        order = await self.exchange_client.place_order(
            symbol=symbol,
            side=side,
            amount=amount,
        )

        return {
            "executed": True,
            "order": order,
        }

    async def _handle_close_position(self, args: Dict) -> Dict:
        symbol = args["symbol"]

        order = await self.exchange_client.close_position(symbol)
        if order is None:
            return {
                "executed": False,
                "reason": "No position to close",
            }

        return {
            "executed": True,
            "order": order,
        }

    # === ANALYTICS HANDLERS ===

    async def _handle_get_market_analytics(self, args: Dict) -> Dict:
        symbol = args["symbol"]

        ohlcv = await self.exchange_client.get_ohlcv(symbol, "1h", 48)
        prices = [c["close"] for c in ohlcv]

        from src.tools.analytics import compute_market_analytics
        return compute_market_analytics(prices)


# Singleton
_router: Optional[ToolRouter] = None

def get_tool_router() -> ToolRouter:
    global _router
    if _router is None:
        _router = ToolRouter()
    return _router
```

**Critères de validation**:
- [ ] Tous les tools routés vers les bons handlers
- [ ] Validation risk intégrée pour les ordres
- [ ] Gestion des erreurs propre
- [ ] Lazy loading des dépendances

---

## T4.3 - Boucle Multi-Tour avec Tools

### T4.3.1 - Modifier GroqAdapter pour gérer le multi-tour
**Priorité**: HAUTE
**Estimation**: Moyenne

Modifier `src/client/groq_adapter.py` pour supporter les appels de tools itératifs :

```python
async def chat_with_tools(
    self,
    initial_messages: List[Dict[str, Any]],
    tool_router: "ToolRouter",
    max_iterations: int = 5,
    on_token: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Chat avec support des tools, multi-tour.
    Le LLM peut appeler des tools, on exécute, et on renvoie les résultats.

    Args:
        initial_messages: Messages de départ
        tool_router: Router pour exécuter les tools
        max_iterations: Nombre max de tours (sécurité)
        on_token: Callback pour chaque token (streaming)
        on_tool_call: Callback quand un tool est appelé

    Returns:
        {
            "final_response": "texte final du LLM",
            "tool_calls": [...],  # Tous les appels effectués
            "iterations": n,
        }
    """
    messages = list(initial_messages)
    all_tool_calls = []

    for iteration in range(max_iterations):
        response_text = ""
        pending_tool_calls = []

        # Stream la réponse
        for event in self.stream_chat(messages):
            if event["type"] == "token":
                response_text += event["content"]
                if on_token:
                    on_token(event["content"])

            elif event["type"] == "tool_call":
                pending_tool_calls.append(event)
                if on_tool_call:
                    on_tool_call(event["name"], event["arguments"])

            elif event["type"] == "done":
                break

        # Si pas de tool calls, on a terminé
        if not pending_tool_calls:
            return {
                "final_response": response_text,
                "tool_calls": all_tool_calls,
                "iterations": iteration + 1,
            }

        # Ajouter la réponse assistant avec les tool_calls
        assistant_message = {
            "role": "assistant",
            "content": response_text if response_text else None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in pending_tool_calls
            ],
        }
        messages.append(assistant_message)

        # Exécuter les tools et ajouter les résultats
        for tc in pending_tool_calls:
            result = await tool_router.execute(tc["name"], tc["arguments"])

            all_tool_calls.append({
                "name": tc["name"],
                "arguments": tc["arguments"],
                "result": result,
            })

            # Ajouter le résultat comme message tool
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

    # Max iterations atteint
    return {
        "final_response": response_text,
        "tool_calls": all_tool_calls,
        "iterations": max_iterations,
        "warning": "Max iterations reached",
    }
```

**Critères de validation**:
- [ ] Multi-tour fonctionnel
- [ ] Limite de sécurité (max_iterations)
- [ ] Callbacks pour UI
- [ ] Historique des tools appelés

---

### T4.3.2 - Modifier loop.py pour utiliser chat_with_tools
**Priorité**: HAUTE
**Estimation**: Moyenne

Modifier `src/bot/loop.py` :

```python
async def _think(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Envoie le snapshot au LLM avec accès aux tools.
    Le LLM peut appeler des tools pour avoir plus d'infos,
    puis retourne ses décisions.
    """
    import json
    from src.tools.router import get_tool_router
    from src.tools.schemas import ALL_TOOLS

    router = get_tool_router()

    user_content = (
        "Voici le snapshot d'état actuel. Analyse-le et décide des actions.\n"
        "Tu peux utiliser tes tools pour obtenir plus d'informations.\n"
        "Quand tu as assez d'infos, retourne un JSON avec tes décisions.\n\n"
        f"SNAPSHOT:\n```json\n{json.dumps(snapshot, indent=2)}\n```\n\n"
        "Format de réponse attendu:\n"
        "{ \"actions\": [\n"
        "  {\"type\": \"OPEN\", \"symbol\": \"...\", \"side\": \"buy\", \"size_pct_equity\": 0.02},\n"
        "  {\"type\": \"CLOSE\", \"symbol\": \"...\"},\n"
        "  {\"type\": \"HOLD\", \"reason\": \"...\"}\n"
        "]}"
    )

    messages = [{"role": "user", "content": user_content}]

    # Callbacks pour le renderer
    def on_token(token: str):
        if self.renderer:
            self.renderer.write_thought(token)
            self.renderer.refresh()

    def on_tool_call(name: str, args: Dict):
        if self.renderer:
            self.renderer.write_thought(f"\n🔧 Calling {name}({args})...\n")
            self.renderer.add_action("TOOL", name, str(args)[:50])
            self.renderer.refresh()

    # Passer les tools au GroqAdapter
    self.groq.tools = ALL_TOOLS

    # Chat avec tools
    result = await self.groq.chat_with_tools(
        initial_messages=messages,
        tool_router=router,
        max_iterations=5,
        on_token=on_token,
        on_tool_call=on_tool_call,
    )

    # Extraire les actions du résultat
    actions = self._extract_actions_from_text(result["final_response"])

    # Log
    self.memory.log_decision(
        message=f"LLM decision (iterations: {result['iterations']})",
        raw_output=result["final_response"],
        context_snapshot={
            "snapshot": snapshot,
            "tool_calls": result["tool_calls"],
        },
    )

    return actions
```

**Critères de validation**:
- [ ] Tools passés au LLM
- [ ] Multi-tour fonctionnel
- [ ] UI mise à jour pendant les tool calls
- [ ] Logging complet

---

## T4.4 - Optimisations

### T4.4.1 - Cache des résultats de tools
**Priorité**: MOYENNE
**Estimation**: Simple

```python
from functools import lru_cache
from datetime import datetime, timedelta
import asyncio


class ToolCache:
    """Cache simple pour les résultats de tools fréquents"""

    def __init__(self, default_ttl: int = 60) -> None:
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if datetime.now() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)

    def clear(self) -> None:
        self._cache.clear()


# Dans ToolRouter, utiliser le cache pour certains tools
async def _handle_get_market_price(self, args: Dict) -> float:
    symbol = args["symbol"]
    cache_key = f"price:{symbol}"

    # Check cache (TTL: 5s pour les prix)
    cached = self._cache.get(cache_key)
    if cached is not None:
        return cached

    price = await self.exchange_client.get_market_price(symbol)
    self._cache.set(cache_key, price, ttl=5)
    return price
```

**Critères de validation**:
- [ ] Cache avec TTL configurable
- [ ] Invalidation automatique
- [ ] Utilisé pour les données fréquentes

---

### T4.4.2 - Rate limiting des tools
**Priorité**: MOYENNE
**Estimation**: Simple

```python
import time
from collections import defaultdict


class RateLimiter:
    """Rate limiter simple par tool"""

    def __init__(self) -> None:
        self._last_call: Dict[str, float] = defaultdict(float)
        self._limits: Dict[str, float] = {
            "get_google_trends": 2.0,  # 1 call / 2s
            "get_news_sentiment": 1.0,
            "place_order": 1.0,
            "close_position": 1.0,
        }
        self._default_limit = 0.1  # 100ms par défaut

    async def wait(self, tool_name: str) -> None:
        limit = self._limits.get(tool_name, self._default_limit)
        last = self._last_call[tool_name]
        elapsed = time.time() - last

        if elapsed < limit:
            await asyncio.sleep(limit - elapsed)

        self._last_call[tool_name] = time.time()

    def can_call(self, tool_name: str) -> bool:
        limit = self._limits.get(tool_name, self._default_limit)
        last = self._last_call[tool_name]
        return (time.time() - last) >= limit
```

**Critères de validation**:
- [ ] Rate limiting par tool
- [ ] Attente async
- [ ] Configurable

---

## T4.5 - Tests et Validation

### T4.5.1 - Tests unitaires des tools
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `tests/test_tools.py` :

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.tools.router import ToolRouter
from src.tools.schemas import ALL_TOOLS, get_tool_names


@pytest.fixture
def mock_exchange():
    client = AsyncMock()
    client.get_market_price.return_value = 50000.0
    client.get_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "change_24h_pct": 2.5,
    }
    client.get_portfolio_state.return_value = {
        "balance_usdt": 1000.0,
        "positions": [],
    }
    return client


@pytest.fixture
def router(mock_exchange):
    r = ToolRouter()
    r._exchange_client = mock_exchange
    return r


class TestToolSchemas:
    def test_all_tools_have_required_fields(self):
        for tool in ALL_TOOLS:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_names_unique(self):
        names = get_tool_names()
        assert len(names) == len(set(names))


class TestToolRouter:
    @pytest.mark.asyncio
    async def test_get_market_price(self, router, mock_exchange):
        result = await router.execute("get_market_price", {"symbol": "BTC/USDT"})
        assert result["success"]
        assert result["result"] == 50000.0

    @pytest.mark.asyncio
    async def test_unknown_tool(self, router):
        result = await router.execute("unknown_tool", {})
        assert not result["success"]
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_place_order_risk_check(self, router, mock_exchange):
        # Order too large
        result = await router.execute("place_order", {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount_usd": 100,  # > 20 USD limit
        })
        assert not result["result"]["executed"]
        assert "Risk check failed" in result["result"]["reason"]
```

**Critères de validation**:
- [ ] Tests pour tous les tools
- [ ] Tests du risk manager
- [ ] Mocks pour exchange/trends
- [ ] Coverage > 80%

---

## Checklist Finale Phase 4

- [ ] Schemas JSON pour tous les tools
- [ ] ToolRouter fonctionnel
- [ ] Multi-tour avec chat_with_tools
- [ ] Risk validation pour exécutions
- [ ] Cache des résultats fréquents
- [ ] Rate limiting
- [ ] Callbacks pour UI
- [ ] Tests unitaires
- [ ] Intégration avec loop.py

---

## Notes Techniques

### Limites Groq Function Calling
- Llama 3.3 supporte bien le function calling
- Max ~10 tools recommandé
- Bien décrire les tools pour meilleure utilisation

### Sécurité
- Toujours valider les arguments des tools
- Risk check obligatoire avant exécution
- Logging de tous les appels

### Performance
- Cache pour données qui changent peu
- Batch les appels quand possible
- Timeout sur les tools longs

---

## T4.6 - Architecture Modulaire Tools (Plugin System)

> **Objectif**: Assurer que les tools sont modulaires et extensibles.
> Chaque tool implémente `BaseTool` et peut être ajouté/retiré dynamiquement.

### T4.6.1 - Implémenter des Tools avec l'interface BaseTool
**Priorité**: HAUTE
**Estimation**: Moyenne

Modifier les tools pour implémenter l'interface `BaseTool` définie dans Phase 0 :

```python
"""
Exemple d'implémentation d'un tool modulaire.
Tous les tools doivent suivre ce pattern.
"""

from typing import Any, Dict
from src.interfaces import BaseTool, ToolDefinition
from src.tools.registry import register_tool


@register_tool
class GetMarketPriceTool(BaseTool):
    """
    Tool modulaire pour récupérer le prix d'un symbole.
    S'enregistre automatiquement grâce au décorateur.
    """

    def __init__(self) -> None:
        self._exchange = None  # Lazy loading

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_market_price",
            description="Get the current spot price for a trading pair",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair (e.g., 'BTC/USDT')",
                    },
                },
                "required": ["symbol"],
            },
            category="observer",
        )

    async def execute(self, symbol: str) -> Dict[str, Any]:
        """Exécute le tool"""
        from src.container import get_container
        exchange = get_container().exchange
        price = await exchange.get_ticker(symbol)
        return {"symbol": symbol, "price": price["last"]}


@register_tool
class PlaceOrderTool(BaseTool):
    """
    Tool modulaire pour placer un ordre.
    Intègre automatiquement la validation de risque.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="place_order",
            description="Place a market order (validated by risk manager)",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "amount_usd": {"type": "number"},
                },
                "required": ["symbol", "side", "amount_usd"],
            },
            category="agir",
        )

    async def execute(
        self, symbol: str, side: str, amount_usd: float
    ) -> Dict[str, Any]:
        """Exécute avec validation de risque"""
        from src.container import get_container
        from src.interfaces import OrderRequest

        container = get_container()
        exchange = container.exchange
        risk_manager = container.risk_manager

        # Portfolio state pour validation
        balance = await exchange.get_balance()
        portfolio = {"balance_usdt": balance["free"].get("USDT", 0)}

        # Check risk
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            amount=amount_usd,
        )
        market_state = await exchange.get_ticker(symbol)
        check = risk_manager.check_order(order_request, portfolio, market_state)

        if not check.approved:
            return {"executed": False, "reason": check.reason}

        # Execute
        amount = amount_usd / market_state["last"]
        order = await exchange.place_order(symbol, side, amount)
        return {"executed": True, "order": order}


@register_tool
class GetGoogleTrendsTool(BaseTool):
    """Tool modulaire pour Google Trends"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_google_trends",
            description="Get Google Trends interest for keywords",
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to analyze (max 5)",
                    },
                },
                "required": ["keywords"],
            },
            category="observer",
        )

    async def execute(self, keywords: list) -> Dict[str, Any]:
        from src.container import get_container
        provider = get_container().trends_provider
        return await provider.get_interest_over_time(keywords[:5])
```

**Critères de validation**:
- [ ] Tous les tools implémentent BaseTool
- [ ] Auto-enregistrement via décorateur
- [ ] Catégories correctes (observer/reflechir/agir)
- [ ] Documentation des paramètres

---

### T4.6.2 - Modifier le ToolRouter pour utiliser le registre
**Priorité**: HAUTE
**Estimation**: Simple

```python
"""
ToolRouter modulaire utilisant le registre de tools.
"""

from typing import Any, Dict
from src.tools.registry import get_tool_registry


class ModularToolRouter:
    """
    Router utilisant le registre de tools.
    Plus besoin de hardcoder les handlers!
    """

    def __init__(self) -> None:
        self._registry = get_tool_registry()

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Exécute un tool via le registre.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' non trouvé",
                "available_tools": [t.definition.name for t in self._registry.get_all()],
            }

        try:
            result = await tool.execute(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_schemas(self) -> list:
        """Retourne les schemas pour le LLM"""
        return self._registry.get_schemas()

    def get_tools_by_category(self, category: str) -> list:
        """Retourne les tools d'une catégorie"""
        return self._registry.get_by_category(category)

    def list_tools(self) -> Dict[str, list]:
        """Liste les tools par catégorie"""
        return {
            "observer": [t.definition.name for t in self.get_tools_by_category("observer")],
            "reflechir": [t.definition.name for t in self.get_tools_by_category("reflechir")],
            "agir": [t.definition.name for t in self.get_tools_by_category("agir")],
        }
```

**Critères de validation**:
- [ ] Utilise le registre de tools
- [ ] Plus de handlers hardcodés
- [ ] Schemas générés automatiquement
- [ ] Filtrage par catégorie

---

### T4.6.3 - Ajouter un nouveau tool (exemple de modularité)
**Priorité**: MOYENNE
**Estimation**: Simple

Exemple montrant comment ajouter un nouveau tool sans modifier le router :

```python
"""
Nouveau tool custom - exemple de modularité.
Il suffit de créer une classe, elle sera auto-enregistrée.
"""

from src.interfaces import BaseTool, ToolDefinition
from src.tools.registry import register_tool


@register_tool
class GetFearGreedIndexTool(BaseTool):
    """
    Tool custom pour récupérer le Fear & Greed Index.
    Démontre l'extensibilité du système.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_fear_greed_index",
            description="Get the current Crypto Fear & Greed Index (0-100)",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            category="observer",
        )

    async def execute(self) -> Dict[str, Any]:
        """Fetch le Fear & Greed Index depuis l'API alternative.me"""
        import aiohttp

        url = "https://api.alternative.me/fng/"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()

        if data.get("data"):
            fng = data["data"][0]
            return {
                "value": int(fng["value"]),
                "classification": fng["value_classification"],
                "timestamp": fng["timestamp"],
            }
        return {"error": "Failed to fetch Fear & Greed Index"}
```

**Comment ajouter un nouveau tool:**
1. Créer une classe qui hérite de `BaseTool`
2. Ajouter le décorateur `@register_tool`
3. Définir `definition` (nom, description, paramètres, catégorie)
4. Implémenter `execute(**kwargs)`
5. C'est tout! Le tool est automatiquement disponible.

**Critères de validation**:
- [ ] Nouveau tool fonctionnel
- [ ] Pas de modification du router
- [ ] Auto-enregistré au démarrage
- [ ] Visible dans les schemas LLM

---

### T4.6.4 - Configuration des tools activés
**Priorité**: BASSE
**Estimation**: Simple

```python
"""
Configuration pour activer/désactiver des tools.
Permet de contrôler quels tools sont disponibles pour le LLM.
"""

from typing import List, Optional
from src.config import get_config


class ToolConfig:
    """Configuration des tools activés"""

    def __init__(self) -> None:
        self._enabled: Optional[List[str]] = None
        self._disabled: List[str] = []

    def enable_only(self, tool_names: List[str]) -> None:
        """Active uniquement les tools spécifiés"""
        self._enabled = tool_names

    def disable(self, tool_names: List[str]) -> None:
        """Désactive les tools spécifiés"""
        self._disabled.extend(tool_names)

    def is_enabled(self, tool_name: str) -> bool:
        """Vérifie si un tool est activé"""
        if self._enabled is not None:
            return tool_name in self._enabled
        return tool_name not in self._disabled


# Usage dans le router
class ConfigurableToolRouter(ModularToolRouter):
    """Router avec support de configuration"""

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        super().__init__()
        self._config = config or ToolConfig()

    def get_schemas(self) -> list:
        """Retourne uniquement les schemas des tools activés"""
        all_schemas = super().get_schemas()
        return [
            s for s in all_schemas
            if self._config.is_enabled(s["function"]["name"])
        ]

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        """Vérifie si le tool est activé avant exécution"""
        if not self._config.is_enabled(tool_name):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' est désactivé",
            }
        return await super().execute(tool_name, arguments)


# Configuration via .env (optionnel)
def load_tool_config() -> ToolConfig:
    """Charge la config des tools depuis l'environnement"""
    import os
    config = ToolConfig()

    disabled = os.getenv("DISABLED_TOOLS", "")
    if disabled:
        config.disable(disabled.split(","))

    return config
```

**Critères de validation**:
- [ ] Activation/désactivation par tool
- [ ] Config via environnement
- [ ] Schemas filtrés pour LLM
- [ ] Exécution bloquée si désactivé
