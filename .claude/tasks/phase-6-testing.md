# Phase 6 - Tests, Hardening & Deployment

> **Objectif**: Finaliser le bot avec des tests complets, le durcissement de sécurité, et la préparation au déploiement.

## Statut Global
- [ ] Phase complète

## Dépendances
- Toutes les phases précédentes (0-5) complètes

---

## T6.1 - Suite de Tests Unitaires

### T6.1.1 - Tests du module config
**Priorité**: HAUTE
**Estimation**: Simple

Créer `tests/test_config.py` :

```python
import pytest
import os
from unittest.mock import patch

from src.config import Config, load_config, get_config


class TestConfig:
    def test_load_config_defaults(self):
        """Test les valeurs par défaut"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config()

            assert cfg.exchange_id == "mexc"
            assert cfg.paper_trading == True
            assert cfg.base_currency == "USDT"
            assert cfg.max_order_usd == 20.0
            assert cfg.max_equity_pct == 0.05

    def test_load_config_from_env(self):
        """Test le chargement depuis les variables d'env"""
        env = {
            "GROQ_API_KEY": "test_key",
            "EXCHANGE_ID": "bybit",
            "PAPER_TRADING": "false",
            "MAX_ORDER_USD": "50.0",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()

            assert cfg.groq_api_key == "test_key"
            assert cfg.exchange_id == "bybit"
            assert cfg.paper_trading == False
            assert cfg.max_order_usd == 50.0

    def test_get_config_singleton(self):
        """Test que get_config retourne un singleton"""
        import src.config as config_module
        config_module.config = None  # Reset

        cfg1 = get_config()
        cfg2 = get_config()

        assert cfg1 is cfg2
```

**Critères de validation**:
- [ ] Tests des valeurs par défaut
- [ ] Tests du chargement env
- [ ] Tests du singleton

---

### T6.1.2 - Tests du module memory (SQLite)
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `tests/test_memory.py` :

```python
import pytest
import tempfile
import os

from src.bot.memory import BotMemory


@pytest.fixture
def memory():
    """Crée une base de données temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    mem = BotMemory(db_path)
    yield mem

    # Cleanup
    mem.conn.close()
    os.unlink(db_path)


class TestBotMemory:
    def test_init_creates_schema(self, memory):
        """Test que le schéma est créé à l'initialisation"""
        cursor = memory.conn.cursor()

        # Vérifier les tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "trades" in tables
        assert "logs" in tables
        assert "config" in tables

    def test_log_info(self, memory):
        """Test le logging INFO"""
        memory.log_info("Test message", context={"key": "value"})

        cursor = memory.conn.cursor()
        cursor.execute("SELECT * FROM logs WHERE level = 'INFO'")
        row = cursor.fetchone()

        assert row is not None
        assert "Test message" in row[3]  # message column

    def test_log_error(self, memory):
        """Test le logging ERROR"""
        memory.log_error("Error message", context={})

        cursor = memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM logs WHERE level = 'ERROR'")
        count = cursor.fetchone()[0]

        assert count == 1

    def test_log_decision(self, memory):
        """Test le logging DECISION avec contexte"""
        memory.log_decision(
            message="Buy BTC",
            raw_output="<thinking>...</thinking>",
            context_snapshot={"portfolio": {}, "market": {}},
        )

        cursor = memory.conn.cursor()
        cursor.execute("SELECT * FROM logs WHERE level = 'DECISION'")
        row = cursor.fetchone()

        assert row is not None
        assert "_raw_llm_output" in row[4]  # context_snapshot JSON

    def test_log_trade_open(self, memory):
        """Test l'enregistrement d'un trade"""
        order = {
            "id": "123",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
        }
        memory.log_trade_open(order, snapshot={}, action={})

        cursor = memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM logs WHERE level = 'TRADE_OPEN'")
        count = cursor.fetchone()[0]

        assert count == 1
```

**Critères de validation**:
- [ ] Tests création schéma
- [ ] Tests logging tous niveaux
- [ ] Tests avec fixture DB temporaire

---

### T6.1.3 - Tests du Risk Manager
**Priorité**: CRITIQUE
**Estimation**: Haute

Créer `tests/test_risk.py` :

```python
import pytest
from src.tools.risk import RiskManager, RiskLimits, check_risk


@pytest.fixture
def risk_manager():
    """RiskManager avec limites de test"""
    limits = RiskLimits(
        max_order_usd=20.0,
        max_equity_pct=0.05,
        min_order_usd=1.0,
        max_daily_trades=10,
        max_daily_loss_usd=50.0,
    )
    return RiskManager(limits=limits)


class TestRiskManagerBasic:
    def test_order_within_limits(self, risk_manager):
        """Test ordre valide"""
        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="BTC/USDT",
        )
        assert ok
        assert reason == "OK"

    def test_order_too_large(self, risk_manager):
        """Test ordre trop grand (USD)"""
        ok, reason = risk_manager.check_order(
            amount_usd=50.0,  # > 20 USD limit
            balance_usd=1000.0,
            symbol="BTC/USDT",
        )
        assert not ok
        assert "too large" in reason.lower()

    def test_order_too_small(self, risk_manager):
        """Test ordre trop petit"""
        ok, reason = risk_manager.check_order(
            amount_usd=0.5,  # < 1 USD min
            balance_usd=1000.0,
            symbol="BTC/USDT",
        )
        assert not ok
        assert "too small" in reason.lower()

    def test_order_exceeds_equity_pct(self, risk_manager):
        """Test ordre dépasse % equity"""
        ok, reason = risk_manager.check_order(
            amount_usd=15.0,  # 15% of 100
            balance_usd=100.0,  # 5% max = 5 USD
            symbol="BTC/USDT",
        )
        assert not ok
        assert "equity" in reason.lower()

    def test_zero_balance(self, risk_manager):
        """Test avec balance zero"""
        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=0.0,
            symbol="BTC/USDT",
        )
        assert not ok

    def test_insufficient_balance(self, risk_manager):
        """Test avec balance insuffisante"""
        ok, reason = risk_manager.check_order(
            amount_usd=15.0,
            balance_usd=10.0,
            symbol="BTC/USDT",
        )
        assert not ok
        assert "insufficient" in reason.lower()


class TestRiskManagerDailyLimits:
    def test_daily_trade_limit(self, risk_manager):
        """Test limite de trades quotidiens"""
        # Simuler 10 trades
        for i in range(10):
            risk_manager.session.trades_count = i + 1

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="BTC/USDT",
        )
        assert not ok
        assert "daily" in reason.lower()

    def test_daily_loss_halt(self, risk_manager):
        """Test arrêt sur perte quotidienne"""
        risk_manager.session.total_pnl_usd = -60.0  # > 50 USD limit

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="BTC/USDT",
        )
        assert not ok
        assert risk_manager.session.is_halted


class TestRiskManagerPositions:
    def test_max_positions(self, risk_manager):
        """Test limite de positions ouvertes"""
        positions = {
            f"TOKEN{i}/USDT": {"value_usdt": 10}
            for i in range(5)  # 5 positions
        }

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="NEWTOKEN/USDT",  # Nouvelle position
            current_positions=positions,
        )
        assert not ok
        assert "position" in reason.lower()

    def test_existing_position_can_add(self, risk_manager):
        """Test ajout à position existante"""
        positions = {
            "BTC/USDT": {"value_usdt": 10},
        }

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="BTC/USDT",  # Position existante
            current_positions=positions,
        )
        assert ok  # OK car position existante


class TestRiskManagerLiquidity:
    def test_low_liquidity_rejected(self, risk_manager):
        """Test rejet pour liquidité insuffisante"""
        liquidity = {
            "tradeable": False,
            "ask_liquidity_usdt": 50,
        }

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="ILLIQUID/USDT",
            liquidity_info=liquidity,
        )
        assert not ok
        assert "liquidity" in reason.lower()

    def test_high_spread_rejected(self, risk_manager):
        """Test rejet pour spread trop élevé"""
        liquidity = {
            "tradeable": True,
            "ask_liquidity_usdt": 10000,
            "spread_pct": 5.0,  # > 2% limit
        }

        ok, reason = risk_manager.check_order(
            amount_usd=10.0,
            balance_usd=1000.0,
            symbol="SPREAD/USDT",
            liquidity_info=liquidity,
        )
        assert not ok
        assert "spread" in reason.lower()


class TestRiskManagerHelpers:
    def test_calculate_safe_order_size(self, risk_manager):
        """Test calcul de taille sûre"""
        size = risk_manager.calculate_safe_order_size(
            balance_usd=1000.0,
            target_pct=0.02,  # 2%
        )

        assert size == 20.0  # 2% of 1000, capped at max

    def test_calculate_safe_order_size_capped(self, risk_manager):
        """Test que la taille est cappée"""
        size = risk_manager.calculate_safe_order_size(
            balance_usd=10000.0,
            target_pct=0.10,  # 10% = 1000 USD
        )

        assert size == 20.0  # Capped at max_order_usd

    def test_get_risk_level(self, risk_manager):
        """Test évaluation du niveau de risque"""
        from src.tools.risk import RiskLevel

        # Pas de perte
        assert risk_manager.get_risk_level() == RiskLevel.LOW

        # Perte modérée
        risk_manager.session.total_pnl_usd = -30  # 60% of limit
        assert risk_manager.get_risk_level() == RiskLevel.MEDIUM

        # Perte importante
        risk_manager.session.total_pnl_usd = -45  # 90% of limit
        assert risk_manager.get_risk_level() == RiskLevel.HIGH


class TestCheckRiskFunction:
    """Tests de la fonction simple check_risk"""

    def test_valid_order(self):
        ok, reason = check_risk(10.0, 1000.0)
        assert ok

    def test_invalid_order(self):
        ok, reason = check_risk(50.0, 100.0)
        assert not ok
```

**Critères de validation**:
- [ ] Tests toutes les règles
- [ ] Tests limites journalières
- [ ] Tests positions
- [ ] Tests liquidité
- [ ] Coverage > 90%

---

### T6.1.4 - Tests du module market
**Priorité**: HAUTE
**Estimation**: Haute

Créer `tests/test_market.py` :

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.tools.market import (
    ExchangeClient,
    PaperExchangeClient,
    create_exchange_client,
)


@pytest.fixture
def mock_ccxt_exchange():
    """Mock d'un exchange CCXT"""
    exchange = AsyncMock()
    exchange.fetch_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49990.0,
        "ask": 50010.0,
        "baseVolume": 1000,
        "quoteVolume": 50000000,
        "percentage": 2.5,
        "high": 51000,
        "low": 49000,
    }
    exchange.fetch_balance.return_value = {
        "total": {"USDT": 1000.0, "BTC": 0.01},
        "free": {"USDT": 1000.0, "BTC": 0.01},
        "used": {},
    }
    exchange.create_market_order.return_value = {
        "id": "123",
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "amount": 0.001,
        "average": 50000,
        "cost": 50,
        "status": "filled",
    }
    return exchange


class TestExchangeClient:
    @pytest.mark.asyncio
    async def test_get_market_price(self, mock_ccxt_exchange):
        client = ExchangeClient()
        client._exchange = mock_ccxt_exchange

        price = await client.get_market_price("BTC/USDT")

        assert price == 50000.0
        mock_ccxt_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_ccxt_exchange):
        client = ExchangeClient()
        client._exchange = mock_ccxt_exchange

        ticker = await client.get_ticker("BTC/USDT")

        assert ticker["last"] == 50000.0
        assert ticker["change_24h_pct"] == 2.5

    @pytest.mark.asyncio
    async def test_get_portfolio_state(self, mock_ccxt_exchange):
        client = ExchangeClient()
        client._exchange = mock_ccxt_exchange

        # Mock paper trading = False
        with patch("src.tools.market.get_config") as mock_cfg:
            mock_cfg.return_value.paper_trading = False
            mock_cfg.return_value.base_currency = "USDT"

            portfolio = await client.get_portfolio_state()

        assert portfolio["balance_usdt"] == 1000.0
        assert portfolio["mode"] == "live"


class TestPaperExchangeClient:
    @pytest.fixture
    def paper_client(self, mock_ccxt_exchange):
        client = PaperExchangeClient(initial_balance_usdt=1000.0)
        client._exchange = mock_ccxt_exchange
        return client

    @pytest.mark.asyncio
    async def test_initial_balance(self, paper_client):
        balance = await paper_client.get_balance()
        assert balance["free"]["USDT"] == 1000.0

    @pytest.mark.asyncio
    async def test_place_buy_order(self, paper_client):
        order = await paper_client.place_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
        )

        assert order["status"] == "filled"
        assert order["side"] == "buy"
        assert "fee" in order

        # Vérifier la balance mise à jour
        balance = await paper_client.get_balance()
        assert balance["free"]["USDT"] < 1000.0
        assert balance["free"]["BTC"] == 0.001

    @pytest.mark.asyncio
    async def test_place_sell_order(self, paper_client):
        # D'abord acheter
        await paper_client.place_order("BTC/USDT", "buy", 0.001)

        # Puis vendre
        order = await paper_client.place_order("BTC/USDT", "sell", 0.001)

        assert order["status"] == "filled"
        assert order["side"] == "sell"

        # Vérifier qu'il n'y a plus de BTC
        balance = await paper_client.get_balance()
        assert balance["free"].get("BTC", 0) == 0

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, paper_client):
        with pytest.raises(ValueError, match="Insufficient"):
            await paper_client.place_order("BTC/USDT", "buy", 100)  # Trop cher

    @pytest.mark.asyncio
    async def test_paper_summary(self, paper_client):
        # Faire quelques trades
        await paper_client.place_order("BTC/USDT", "buy", 0.001)

        summary = await paper_client.get_paper_summary()

        assert "initial_balance" in summary
        assert "total_equity" in summary
        assert "stats" in summary
        assert summary["stats"]["total_trades"] == 1


class TestCreateExchangeClient:
    def test_creates_paper_client_when_paper_mode(self):
        with patch("src.tools.market.get_config") as mock_cfg:
            mock_cfg.return_value.paper_trading = True

            client = create_exchange_client()

            assert isinstance(client, PaperExchangeClient)

    def test_creates_real_client_when_live_mode(self):
        with patch("src.tools.market.get_config") as mock_cfg:
            mock_cfg.return_value.paper_trading = False

            client = create_exchange_client()

            assert isinstance(client, ExchangeClient)
            assert not isinstance(client, PaperExchangeClient)
```

**Critères de validation**:
- [ ] Tests ExchangeClient avec mocks
- [ ] Tests PaperExchangeClient complets
- [ ] Tests factory
- [ ] Tests async corrects

---

## T6.2 - Tests d'Intégration

### T6.2.1 - Tests end-to-end de la boucle
**Priorité**: HAUTE
**Estimation**: Haute

Créer `tests/test_integration.py` :

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.loop import TradingBotLoop
from src.client.groq_adapter import GroqAdapter


@pytest.fixture
def mock_groq():
    """Mock du client Groq"""
    adapter = MagicMock(spec=GroqAdapter)

    # Simuler une réponse LLM simple
    def mock_stream(*args, **kwargs):
        yield {"type": "token", "content": "Analyzing market... "}
        yield {"type": "token", "content": '{"actions": [{"type": "HOLD", "reason": "No opportunity"}]}'}
        yield {"type": "done"}

    adapter.stream_chat = mock_stream
    return adapter


@pytest.fixture
def mock_exchange():
    """Mock de l'exchange"""
    exchange = AsyncMock()
    exchange.get_market_snapshot.return_value = {
        "tickers": {"BTC/USDT": {"last": 50000}},
    }
    exchange.get_portfolio_state.return_value = {
        "balance_usdt": 1000.0,
        "positions": [],
    }
    return exchange


class TestTradingBotLoopIntegration:
    @pytest.mark.asyncio
    async def test_single_loop_iteration(self, mock_groq, mock_exchange):
        """Test une itération complète de la boucle"""
        with patch("src.bot.loop.get_exchange_client", return_value=mock_exchange):
            with patch("src.bot.loop.get_trend_analyzer") as mock_trends:
                mock_trends.return_value.get_trend_snapshot = AsyncMock(return_value={})

                loop = TradingBotLoop(groq_client=mock_groq)

                # Exécuter une itération
                snapshot = await loop._observe()
                actions = await loop._think(snapshot)

                assert snapshot is not None
                assert isinstance(actions, list)

    @pytest.mark.asyncio
    async def test_loop_handles_errors_gracefully(self, mock_groq, mock_exchange):
        """Test que la boucle gère les erreurs sans crash"""
        mock_exchange.get_portfolio_state.side_effect = Exception("Network error")

        with patch("src.bot.loop.get_exchange_client", return_value=mock_exchange):
            loop = TradingBotLoop(groq_client=mock_groq)

            # La boucle ne devrait pas crash
            try:
                snapshot = await loop._observe()
            except Exception:
                pass  # Expected

            # Le bot devrait continuer (logging de l'erreur)

    @pytest.mark.asyncio
    async def test_action_execution_flow(self, mock_groq, mock_exchange):
        """Test le flux complet d'exécution d'une action"""
        # Simuler une décision d'achat
        def mock_stream_buy(*args, **kwargs):
            yield {"type": "token", "content": '{"actions": ['}
            yield {"type": "token", "content": '{"type": "OPEN", "symbol": "BTC/USDT", "side": "buy", "size_pct_equity": 0.02}'}
            yield {"type": "token", "content": ']}'}
            yield {"type": "done"}

        mock_groq.stream_chat = mock_stream_buy

        with patch("src.bot.loop.get_exchange_client", return_value=mock_exchange):
            with patch("src.bot.loop.get_risk_manager") as mock_risk:
                mock_risk.return_value.check_order.return_value = (True, "OK")
                mock_risk.return_value.calculate_safe_order_size.return_value = 20.0

                loop = TradingBotLoop(groq_client=mock_groq)

                snapshot = {"portfolio": {"balance_usdt": 1000, "positions": []}}
                actions = [
                    {"type": "OPEN", "symbol": "BTC/USDT", "side": "buy", "size_pct_equity": 0.02}
                ]

                await loop._act(snapshot, actions)

                # Vérifier que place_order a été appelé
                mock_exchange.place_order.assert_called_once()
```

**Critères de validation**:
- [ ] Test itération complète
- [ ] Test gestion des erreurs
- [ ] Test flux d'exécution
- [ ] Mocks correctement configurés

---

## T6.3 - Hardening de Sécurité

### T6.3.1 - Audit de sécurité du code
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Checklist de sécurité :

```markdown
## Audit de Sécurité - Checklist

### 1. Gestion des Secrets
- [ ] API keys jamais dans le code
- [ ] .env dans .gitignore
- [ ] Pas de logging des secrets
- [ ] Validation des secrets au démarrage

### 2. Validation des Entrées
- [ ] Validation des symboles (format correct)
- [ ] Validation des montants (positifs, dans les limites)
- [ ] Sanitization des inputs LLM
- [ ] Pas d'injection SQL possible

### 3. Limites de Sécurité
- [ ] Hard limits incontournables dans le code
- [ ] Pas de bypass possible via config
- [ ] Rate limiting sur les API calls
- [ ] Timeout sur toutes les opérations réseau

### 4. Gestion des Erreurs
- [ ] Pas de stack traces exposées
- [ ] Logging sécurisé (pas de secrets)
- [ ] Fallback gracieux en cas d'erreur
- [ ] Halt automatique sur erreurs critiques

### 5. Audit du Code LLM
- [ ] System prompt ne peut pas être modifié
- [ ] Actions limitées aux tools autorisés
- [ ] Validation des outputs LLM avant exécution
- [ ] Logging de toutes les décisions
```

**Actions à implémenter** :

```python
# src/security.py

import re
from typing import Optional


def validate_symbol(symbol: str) -> bool:
    """Valide le format d'un symbole de trading"""
    pattern = r"^[A-Z0-9]{2,10}/[A-Z]{3,4}$"
    return bool(re.match(pattern, symbol.upper()))


def validate_amount(amount: float, min_val: float = 0, max_val: float = float("inf")) -> bool:
    """Valide un montant"""
    return isinstance(amount, (int, float)) and min_val <= amount <= max_val


def sanitize_log_message(message: str) -> str:
    """Supprime les informations sensibles des logs"""
    # Masquer les clés API potentielles
    patterns = [
        (r"(api[_-]?key[=:]\s*)['\"]?[\w-]+['\"]?", r"\1[REDACTED]"),
        (r"(secret[=:]\s*)['\"]?[\w-]+['\"]?", r"\1[REDACTED]"),
        (r"(password[=:]\s*)['\"]?[\w-]+['\"]?", r"\1[REDACTED]"),
    ]
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
    return message


def validate_llm_action(action: dict) -> tuple[bool, str]:
    """Valide une action proposée par le LLM"""
    allowed_types = {"OPEN", "CLOSE", "HOLD", "ADJUST"}
    allowed_sides = {"buy", "sell"}

    action_type = action.get("type", "").upper()
    if action_type not in allowed_types:
        return False, f"Invalid action type: {action_type}"

    if action_type in ("OPEN", "ADJUST"):
        symbol = action.get("symbol", "")
        if not validate_symbol(symbol):
            return False, f"Invalid symbol: {symbol}"

        side = action.get("side", "").lower()
        if side and side not in allowed_sides:
            return False, f"Invalid side: {side}"

        size_pct = action.get("size_pct_equity", 0)
        if not validate_amount(size_pct, 0, 0.10):  # Max 10%
            return False, f"Invalid size_pct: {size_pct}"

    return True, "OK"
```

**Critères de validation**:
- [ ] Checklist complète vérifiée
- [ ] Fonctions de validation implémentées
- [ ] Tests de sécurité passants

---

### T6.3.2 - Hardening de la configuration
**Priorité**: HAUTE
**Estimation**: Simple

```python
# src/config.py - Ajouter la validation

def validate_config(cfg: Config) -> list[str]:
    """Valide la configuration et retourne les erreurs"""
    errors = []

    # API key obligatoire
    if not cfg.groq_api_key:
        errors.append("GROQ_API_KEY is required")

    # Limites de sécurité
    if cfg.max_order_usd > 100:
        errors.append("max_order_usd > 100 is dangerous")

    if cfg.max_equity_pct > 0.10:
        errors.append("max_equity_pct > 10% is dangerous")

    # Mode live nécessite plus de config
    if not cfg.paper_trading:
        if not cfg.exchange_api_key:
            errors.append("EXCHANGE_API_KEY required for live trading")
        if not cfg.exchange_api_secret:
            errors.append("EXCHANGE_API_SECRET required for live trading")

    return errors


def load_config_safe() -> tuple[Optional[Config], list[str]]:
    """Charge et valide la configuration"""
    cfg = load_config()
    errors = validate_config(cfg)
    if errors:
        return None, errors
    return cfg, []
```

---

## T6.4 - Documentation

### T6.4.1 - Créer README.md complet
**Priorité**: HAUTE
**Estimation**: Moyenne

Structure du README :

```markdown
# OtterTrend 🦦

Bot de trading autonome spécialisé SocialFi/Crypto trends.

## Features
- Analyse Google Trends et sentiment news
- Trading autonome via MEXC
- Risk management robuste
- Mode paper trading
- UI Rich en temps réel

## Quick Start
1. Clone et install
2. Configure .env
3. python main.py

## Configuration
[Détails des variables d'env]

## Architecture
[Diagramme et explication des modules]

## Risk Management
[Explication des limites]

## Development
[Guide de contribution]

## License
MIT
```

**Critères de validation**:
- [ ] Installation claire
- [ ] Configuration documentée
- [ ] Exemples d'utilisation
- [ ] Section sécurité

---

### T6.4.2 - Documenter les modules avec docstrings
**Priorité**: MOYENNE
**Estimation**: Moyenne

Tous les modules publics doivent avoir :
- Docstring de module en haut du fichier
- Docstring pour chaque classe
- Docstring pour chaque méthode publique
- Type hints complets

---

## T6.5 - Préparation au Déploiement

### T6.5.1 - Créer le Dockerfile
**Priorité**: MOYENNE
**Estimation**: Simple

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY main.py .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Default command
CMD ["python", "main.py"]
```

---

### T6.5.2 - Créer docker-compose.yml
**Priorité**: MOYENNE
**Estimation**: Simple

```yaml
# docker-compose.yml
version: '3.8'

services:
  ottertrend:
    build: .
    env_file: .env
    volumes:
      - ./bot_data.db:/app/bot_data.db
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

### T6.5.3 - Script de démarrage avec checks
**Priorité**: HAUTE
**Estimation**: Simple

```python
#!/usr/bin/env python3
# scripts/start.py
"""Script de démarrage avec vérifications"""

import sys
from src.config import load_config_safe
from src.tools.risk import LiveReadinessValidator


def preflight_checks():
    """Vérifications avant démarrage"""
    print("🦦 OtterTrend - Preflight Checks")
    print("=" * 40)

    # 1. Config
    print("\n1. Checking configuration...")
    cfg, errors = load_config_safe()
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        return False
    print("   ✅ Configuration OK")

    # 2. Dependencies
    print("\n2. Checking dependencies...")
    try:
        import groq
        import ccxt
        import rich
        print("   ✅ All dependencies installed")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return False

    # 3. API connectivity
    print("\n3. Checking API connectivity...")
    # TODO: Test Groq API
    # TODO: Test Exchange API
    print("   ⚠️ API checks skipped (implement later)")

    # 4. Paper trading check (if live mode)
    if not cfg.paper_trading:
        print("\n4. Live mode readiness check...")
        # TODO: Load paper trading summary and validate
        print("   ⚠️ Live mode checks skipped")

    print("\n" + "=" * 40)
    print("✅ All preflight checks passed!")
    return True


if __name__ == "__main__":
    if not preflight_checks():
        print("\n❌ Preflight checks failed. Fix errors and retry.")
        sys.exit(1)

    print("\nStarting OtterTrend...")
    import main
    sys.exit(main.main())
```

---

## Checklist Finale Phase 6

### Tests
- [ ] Tests unitaires config
- [ ] Tests unitaires memory
- [ ] Tests unitaires risk (90%+ coverage)
- [ ] Tests unitaires market
- [ ] Tests d'intégration boucle
- [ ] Tous les tests passent

### Sécurité
- [ ] Audit checklist complète
- [ ] Fonctions de validation
- [ ] Sanitization des logs
- [ ] Hard limits vérifiés

### Documentation
- [ ] README.md complet
- [ ] Docstrings sur tous les modules
- [ ] Exemples d'utilisation
- [ ] Guide de contribution

### Déploiement
- [ ] Dockerfile fonctionnel
- [ ] docker-compose.yml
- [ ] Script de démarrage avec checks
- [ ] Configuration production

### Validation Finale
- [ ] Bot tourne 24h en paper trading sans crash
- [ ] Toutes les features fonctionnent
- [ ] Logs propres et utiles
- [ ] Performance acceptable

---

## T6.6 - Tests de résilience « Envoy-like » (réseau CEX)

### T6.6.1 - Chaos tests circuit breaker / retries / hedging
**Priorité**: HAUTE  \
**Estimation**: Moyenne

- Simuler erreurs 429/5xx/timeout et latence >800ms pour vérifier :
  - ouverture/fermeture du circuit breaker par route (état exposé dans les logs/traces),
  - retries avec backoff+jitter respectant les budgets rate limit,
  - hedging qui annule la requête lente.
- Tester la demi-ouverture (1-3 probes) et la propagation d'erreurs typées vers le risk manager.

**Critères de validation**:
- [ ] Tests unitaires ou d'intégration qui trippent le breaker et valident les transitions.
- [ ] Logs/traces contiennent les événements breaker/hedging.
- [ ] Aucun dépassement des quotas configurés.

---

### T6.6.2 - Charge contrôlée & adaptive concurrency
**Priorité**: HAUTE  \
**Estimation**: Moyenne

- Test de charge (locust/pytest-asyncio) sur `get_tickers`/`fetch_order_book` pour vérifier :
  - adaptation automatique de la concurrence en fonction de la latence (pattern Envoy 2025),
  - load shedding quand la file dépasse le seuil, avec code explicite.
- Vérifier la priorité (orders > account > market bulk) sous contention.

**Critères de validation**:
- [ ] Concurrency max diminue quand latence augmente, puis remonte après cooldown.
- [ ] Shed events comptabilisés et non bloquants pour les ordres.
- [ ] Mesure p95 latence reste sous le seuil cible.

---

### T6.6.3 - Observabilité OpenTelemetry
**Priorité**: HAUTE  \
**Estimation**: Simple

- Tests qui valident l'émission de traces OTLP et de métriques Prometheus :
  - `cex_http_latency_ms`, `cex_shed_total`, `cex_breaker_open_total`, connexions actives,
  - corrélation trace_id/span_id dans les logs JSON.
- Vérifier que l'export peut être activé/désactivé via `config` et que les labels incluent endpoint/symbol.

**Critères de validation**:
- [ ] Export OTLP activable et observable dans un collecteur local (ou mock OTLP).
- [ ] Metrics exposées et nommées correctement.
- [ ] Logs corrélés aux actions LLM.

---

## Notes de Déploiement

### Recommandations Production
1. Utiliser un VPS stable (pas de shutdown inattendu)
2. Monitorer les logs en continu
3. Configurer des alertes (email/Telegram)
4. Backup régulier de la DB
5. Commencer avec des micro-montants

### Checklist Pre-Live
1. ✅ 50+ trades en paper trading
2. ✅ Win rate > 40%
3. ✅ Drawdown max < 10%
4. ✅ Pas de bugs critiques
5. ✅ API keys configurées
6. ✅ Limites de risque vérifiées
