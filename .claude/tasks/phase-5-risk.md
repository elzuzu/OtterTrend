# Phase 5 - Paper Trading & Risk Management

> **Objectif**: Implémenter un système de risk management robuste et finaliser le mode paper trading pour tester le bot en toute sécurité.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0-4 complètes
- Particulièrement: Phase 1 (market client), Phase 4 (tool router)

**Paramètres de base** :
* Lire les seuils via `.env` (`RISK_MAX_TRADE_USD`, `RISK_MAX_TRADE_PCT_EQUITY`, `RISK_MIN_LIQUIDITY_USD`, `RISK_LOW_LIQUIDITY_CAP_USD`).
* Ajouter `estimate_pair_liquidity(symbol)` : refuse les paires sous le seuil de volume 24h et plafonne les ordres bas-volumes.

---

## T5.1 - Risk Manager Complet

### T5.1.1 - Créer src/tools/risk.py avec toutes les règles
**Priorité**: CRITIQUE
**Estimation**: Haute

```python
"""
Risk Manager pour OtterTrend.
Implémente des garde-fous stricts pour éviter les pertes catastrophiques.
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.config import get_config


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Limites de risque configurables"""
    # Limites par ordre
    max_order_usd: float = 20.0           # Max USD par ordre
    max_equity_pct: float = 0.05          # Max 5% de l'equity par ordre
    min_order_usd: float = 1.0            # Min USD par ordre

    # Limites globales
    max_open_positions: int = 5           # Nombre max de positions
    max_total_exposure_pct: float = 0.30  # Max 30% de l'equity exposé
    max_single_asset_pct: float = 0.10    # Max 10% sur un seul asset

    # Limites journalières
    max_daily_trades: int = 50            # Nombre max de trades/jour
    max_daily_loss_pct: float = 0.05      # Stop si perte > 5% en un jour
    max_daily_loss_usd: float = 50.0      # Stop si perte > 50 USD

    # Limites de liquidité
    min_liquidity_usd: float = 1000.0     # Min liquidité pour trader
    max_spread_pct: float = 2.0           # Max spread acceptable

    # Limites de volatilité
    max_volatility: float = 0.10          # Skip si volatilité > 10%

    @classmethod
    def from_config(cls) -> "RiskLimits":
        """Charge les limites depuis la config"""
        cfg = get_config()
        return cls(
            max_order_usd=cfg.max_order_usd,
            max_equity_pct=cfg.max_equity_pct,
        )


@dataclass
class TradingSession:
    """État de la session de trading (quotidien)"""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    trades_count: int = 0
    total_pnl_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    peak_equity_usd: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Gestionnaire de risque central.
    Valide tous les ordres avant exécution.
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits.from_config()
        self.session = TradingSession()
        self._last_order_time: Dict[str, datetime] = {}  # Anti-spam par symbole

    def reset_session(self) -> None:
        """Reset la session (nouveau jour)"""
        self.session = TradingSession()

    def check_session_date(self) -> None:
        """Vérifie si on doit reset la session (nouveau jour)"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.session.date != today:
            self.reset_session()

    # === VALIDATION D'ORDRE ===

    def check_order(
        self,
        amount_usd: float,
        balance_usd: float,
        symbol: str,
        side: str = "buy",
        current_positions: Optional[Dict] = None,
        liquidity_info: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """
        Validation complète d'un ordre.

        Returns:
            (ok, reason) - ok=True si l'ordre peut passer
        """
        self.check_session_date()

        # 1. Vérifier si la session est arrêtée
        if self.session.is_halted:
            return False, f"Trading halted: {self.session.halt_reason}"

        # 2. Vérifier les limites de base
        ok, reason = self._check_basic_limits(amount_usd, balance_usd)
        if not ok:
            return False, reason

        # 3. Vérifier les limites journalières
        ok, reason = self._check_daily_limits()
        if not ok:
            return False, reason

        # 4. Vérifier les positions existantes
        if current_positions:
            ok, reason = self._check_position_limits(
                symbol, amount_usd, balance_usd, current_positions
            )
            if not ok:
                return False, reason

        # 5. Vérifier la liquidité
        if liquidity_info:
            ok, reason = self._check_liquidity(amount_usd, liquidity_info)
            if not ok:
                return False, reason

        # 6. Anti-spam (min 5s entre ordres sur même symbole)
        ok, reason = self._check_rate_limit(symbol)
        if not ok:
            return False, reason

        return True, "OK"

    def _check_basic_limits(
        self,
        amount_usd: float,
        balance_usd: float,
    ) -> Tuple[bool, str]:
        """Vérifie les limites de base par ordre"""
        # Min order
        if amount_usd < self.limits.min_order_usd:
            return False, f"Order too small: ${amount_usd:.2f} < ${self.limits.min_order_usd}"

        # Max order (hard limit)
        if amount_usd > self.limits.max_order_usd:
            return False, f"Order too large: ${amount_usd:.2f} > ${self.limits.max_order_usd}"

        # Balance check
        if balance_usd <= 0:
            return False, "Zero balance"

        # Max equity percentage
        equity_pct = amount_usd / balance_usd
        if equity_pct > self.limits.max_equity_pct:
            return False, f"Exceeds equity limit: {equity_pct:.1%} > {self.limits.max_equity_pct:.1%}"

        # Sufficient balance
        if amount_usd > balance_usd:
            return False, f"Insufficient balance: ${balance_usd:.2f} < ${amount_usd:.2f}"

        return True, "OK"

    def _check_daily_limits(self) -> Tuple[bool, str]:
        """Vérifie les limites journalières"""
        # Max trades per day
        if self.session.trades_count >= self.limits.max_daily_trades:
            return False, f"Daily trade limit reached: {self.limits.max_daily_trades}"

        # Max daily loss (USD)
        if self.session.total_pnl_usd < -self.limits.max_daily_loss_usd:
            self._halt_trading(f"Daily USD loss limit: ${-self.session.total_pnl_usd:.2f}")
            return False, "Daily loss limit reached (USD)"

        return True, "OK"

    def _check_position_limits(
        self,
        symbol: str,
        amount_usd: float,
        balance_usd: float,
        positions: Dict,
    ) -> Tuple[bool, str]:
        """Vérifie les limites de positions"""
        # Max open positions
        if len(positions) >= self.limits.max_open_positions:
            if symbol not in positions:  # Nouvelle position
                return False, f"Max positions reached: {self.limits.max_open_positions}"

        # Max exposure total
        total_exposure = sum(p.get("value_usdt", 0) for p in positions.values())
        total_exposure += amount_usd
        exposure_pct = total_exposure / balance_usd if balance_usd > 0 else 1

        if exposure_pct > self.limits.max_total_exposure_pct:
            return False, f"Total exposure too high: {exposure_pct:.1%}"

        # Max single asset exposure
        current_position = positions.get(symbol, {}).get("value_usdt", 0)
        new_exposure = current_position + amount_usd
        single_pct = new_exposure / balance_usd if balance_usd > 0 else 1

        if single_pct > self.limits.max_single_asset_pct:
            return False, f"Single asset exposure too high: {single_pct:.1%}"

        return True, "OK"

    def _check_liquidity(
        self,
        amount_usd: float,
        liquidity: Dict,
    ) -> Tuple[bool, str]:
        """Vérifie la liquidité disponible"""
        if not liquidity.get("tradeable", False):
            return False, "Asset not tradeable (insufficient liquidity)"

        available = liquidity.get("ask_liquidity_usdt", 0)
        if amount_usd > available * 0.1:  # Max 10% de la liquidité dispo
            return False, f"Order too large for liquidity: ${amount_usd:.2f}"

        spread = liquidity.get("spread_pct", 100)
        if spread > self.limits.max_spread_pct:
            return False, f"Spread too high: {spread:.2f}%"

        return True, "OK"

    def _check_rate_limit(self, symbol: str) -> Tuple[bool, str]:
        """Anti-spam: min 5s entre ordres sur même symbole"""
        now = datetime.now()
        last = self._last_order_time.get(symbol)

        if last and (now - last) < timedelta(seconds=5):
            return False, "Rate limit: wait 5s between orders on same symbol"

        self._last_order_time[symbol] = now
        return True, "OK"

    def _halt_trading(self, reason: str) -> None:
        """Arrête le trading pour la session"""
        self.session.is_halted = True
        self.session.halt_reason = reason

    # === RECORDING ===

    def record_trade(self, pnl_usd: float) -> None:
        """Enregistre un trade dans la session"""
        self.check_session_date()
        self.session.trades_count += 1
        self.session.total_pnl_usd += pnl_usd

        # Check daily loss after trade
        if self.session.total_pnl_usd < -self.limits.max_daily_loss_usd:
            self._halt_trading(f"Daily loss limit reached: ${-self.session.total_pnl_usd:.2f}")

    def update_equity(self, equity_usd: float) -> None:
        """Met à jour le peak et drawdown"""
        self.check_session_date()

        if equity_usd > self.session.peak_equity_usd:
            self.session.peak_equity_usd = equity_usd

        drawdown = self.session.peak_equity_usd - equity_usd
        if drawdown > self.session.max_drawdown_usd:
            self.session.max_drawdown_usd = drawdown

    # === HELPERS ===

    def get_session_stats(self) -> Dict[str, Any]:
        """Retourne les stats de la session"""
        return {
            "date": self.session.date,
            "trades_count": self.session.trades_count,
            "total_pnl_usd": round(self.session.total_pnl_usd, 2),
            "max_drawdown_usd": round(self.session.max_drawdown_usd, 2),
            "is_halted": self.session.is_halted,
            "halt_reason": self.session.halt_reason,
            "remaining_trades": self.limits.max_daily_trades - self.session.trades_count,
        }

    def get_risk_level(self) -> RiskLevel:
        """Évalue le niveau de risque actuel"""
        if self.session.is_halted:
            return RiskLevel.CRITICAL

        loss_ratio = abs(self.session.total_pnl_usd) / self.limits.max_daily_loss_usd

        if loss_ratio > 0.8:
            return RiskLevel.HIGH
        elif loss_ratio > 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def calculate_safe_order_size(
        self,
        balance_usd: float,
        target_pct: float = 0.02,
    ) -> float:
        """Calcule une taille d'ordre sûre"""
        # Base sur le pourcentage demandé
        size = balance_usd * target_pct

        # Cap au max absolu
        size = min(size, self.limits.max_order_usd)

        # Cap au max pourcentage
        max_by_pct = balance_usd * self.limits.max_equity_pct
        size = min(size, max_by_pct)

        # Min size
        size = max(size, self.limits.min_order_usd)

        return round(size, 2)


# === FONCTIONS DE COMMODITÉ ===

def check_risk(amount_usd: float, balance_usd: float) -> Tuple[bool, str]:
    """
    Fonction simple de check (rétrocompatibilité).
    Pour usage avancé, utiliser RiskManager directement.
    """
    manager = RiskManager()
    return manager.check_order(amount_usd, balance_usd, "UNKNOWN/USDT")


# Singleton
_risk_manager: Optional[RiskManager] = None

def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
```

**Critères de validation**:
- [ ] Toutes les règles de risque implémentées
- [ ] Hard limits incontournables
- [ ] Limites journalières avec halt
- [ ] Anti-spam par symbole
- [ ] Stats de session
- [ ] Tests unitaires

---

## T5.2 - Système d'Alertes

### T5.2.1 - Implémenter les alertes de risque
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
"""
Système d'alertes pour les événements de risque.
"""

from typing import Callable, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity: AlertSeverity
    message: str
    timestamp: datetime
    context: dict
    acknowledged: bool = False


class AlertManager:
    """Gestionnaire d'alertes de risque"""

    def __init__(self) -> None:
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._max_alerts = 100

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Ajoute un handler d'alertes (pour UI, logs, etc.)"""
        self._handlers.append(handler)

    def emit(
        self,
        severity: AlertSeverity,
        message: str,
        context: Optional[dict] = None,
    ) -> Alert:
        """Émet une alerte"""
        alert = Alert(
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            context=context or {},
        )

        self._alerts.append(alert)

        # Limiter le nombre d'alertes en mémoire
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        # Notifier les handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass  # Don't let handler errors break the flow

        return alert

    def get_recent(
        self,
        limit: int = 10,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Récupère les alertes récentes"""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def get_unacknowledged(self) -> List[Alert]:
        """Alertes non acquittées"""
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge(self, alert: Alert) -> None:
        """Acquitte une alerte"""
        alert.acknowledged = True

    # === MÉTHODES DE COMMODITÉ ===

    def info(self, message: str, **context) -> Alert:
        return self.emit(AlertSeverity.INFO, message, context)

    def warning(self, message: str, **context) -> Alert:
        return self.emit(AlertSeverity.WARNING, message, context)

    def error(self, message: str, **context) -> Alert:
        return self.emit(AlertSeverity.ERROR, message, context)

    def critical(self, message: str, **context) -> Alert:
        return self.emit(AlertSeverity.CRITICAL, message, context)


# Intégration avec RiskManager
class RiskManagerWithAlerts(RiskManager):
    """RiskManager avec support des alertes"""

    def __init__(self, alerts: Optional[AlertManager] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alerts = alerts or AlertManager()

    def check_order(self, *args, **kwargs) -> Tuple[bool, str]:
        ok, reason = super().check_order(*args, **kwargs)

        if not ok:
            self.alerts.warning(f"Order rejected: {reason}", args=args)

        return ok, reason

    def _halt_trading(self, reason: str) -> None:
        super()._halt_trading(reason)
        self.alerts.critical(f"Trading halted: {reason}")

    def record_trade(self, pnl_usd: float) -> None:
        super().record_trade(pnl_usd)

        if pnl_usd < -10:
            self.alerts.warning(f"Significant loss: ${pnl_usd:.2f}")
        elif pnl_usd > 20:
            self.alerts.info(f"Nice profit: ${pnl_usd:.2f}")
```

**Critères de validation**:
- [ ] Alertes par sévérité
- [ ] Handlers personnalisables
- [ ] Intégration RiskManager
- [ ] Limite mémoire

---

## T5.3 - Finalisation Paper Trading

### T5.3.1 - Améliorer PaperExchangeClient
**Priorité**: HAUTE
**Estimation**: Moyenne

Ajouter les fonctionnalités manquantes au mode paper :

```python
# Ajouts à PaperExchangeClient dans src/tools/market.py

class PaperExchangeClient(ExchangeClient):
    """Version améliorée du paper trading"""

    def __init__(
        self,
        initial_balance_usdt: float = 1000.0,
        simulate_slippage: bool = True,
        simulate_fees: bool = True,
        fee_rate: float = 0.001,  # 0.1%
    ) -> None:
        super().__init__()
        self._initial_balance = initial_balance_usdt
        self._paper_balance: Dict[str, float] = {
            get_config().base_currency: initial_balance_usdt
        }
        self._paper_positions: Dict[str, Dict] = {}
        self._paper_orders: List[Dict] = []
        self._order_counter = 0
        self._simulate_slippage = simulate_slippage
        self._simulate_fees = simulate_fees
        self._fee_rate = fee_rate
        self._trade_history: List[Dict] = []

    def _apply_slippage(self, price: float, side: str) -> float:
        """Simule le slippage (0.05% - 0.2%)"""
        if not self._simulate_slippage:
            return price

        import random
        slippage_pct = random.uniform(0.0005, 0.002)  # 0.05% - 0.2%

        if side == "buy":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def _calculate_fee(self, cost: float) -> float:
        """Calcule les frais"""
        if not self._simulate_fees:
            return 0.0
        return cost * self._fee_rate

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place un ordre simulé avec slippage et frais"""
        cfg = get_config()
        base, quote = symbol.split("/")
        side = side.lower()

        # Récupérer le vrai prix
        real_price = price if price else await self.get_market_price(symbol)

        # Appliquer le slippage
        exec_price = self._apply_slippage(real_price, side)

        # Calculer le coût
        cost = amount * exec_price
        fee = self._calculate_fee(cost)

        if side == "buy":
            total_cost = cost + fee
            quote_balance = self._paper_balance.get(quote, 0.0)

            if total_cost > quote_balance:
                raise ValueError(f"Insufficient {quote}: {quote_balance:.2f} < {total_cost:.2f}")

            # Exécuter
            self._paper_balance[quote] = quote_balance - total_cost
            self._paper_balance[base] = self._paper_balance.get(base, 0.0) + amount

            # Tracker position
            self._update_position(symbol, side, amount, exec_price)

        else:  # sell
            base_balance = self._paper_balance.get(base, 0.0)
            if amount > base_balance:
                amount = base_balance

            if amount <= 0:
                raise ValueError(f"No {base} to sell")

            revenue = amount * exec_price
            fee = self._calculate_fee(revenue)
            net_revenue = revenue - fee

            self._paper_balance[base] = base_balance - amount
            self._paper_balance[quote] = self._paper_balance.get(quote, 0.0) + net_revenue

            # Calculer PnL si on ferme une position
            pnl = self._close_position_partial(symbol, amount, exec_price)

        self._order_counter += 1
        order = {
            "id": f"paper_{self._order_counter}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": exec_price,
            "cost": cost,
            "fee": {"cost": fee, "currency": quote, "rate": self._fee_rate},
            "status": "filled",
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "slippage_pct": (exec_price / real_price - 1) * 100 if side == "buy" else (1 - exec_price / real_price) * 100,
        }

        self._paper_orders.append(order)
        self._trade_history.append({
            **order,
            "pnl": pnl if side == "sell" else None,
        })

        return order

    def _update_position(self, symbol: str, side: str, amount: float, price: float) -> None:
        """Met à jour la position après un achat"""
        if symbol not in self._paper_positions:
            self._paper_positions[symbol] = {
                "amount": 0.0,
                "avg_price": 0.0,
                "cost_basis": 0.0,
                "unrealized_pnl": 0.0,
            }

        pos = self._paper_positions[symbol]
        old_cost = pos["amount"] * pos["avg_price"]
        new_cost = amount * price
        total_amount = pos["amount"] + amount

        if total_amount > 0:
            pos["avg_price"] = (old_cost + new_cost) / total_amount

        pos["amount"] = total_amount
        pos["cost_basis"] = pos["amount"] * pos["avg_price"]

    def _close_position_partial(self, symbol: str, amount: float, price: float) -> float:
        """Ferme partiellement une position et calcule le PnL"""
        if symbol not in self._paper_positions:
            return 0.0

        pos = self._paper_positions[symbol]
        if pos["amount"] <= 0:
            return 0.0

        # Calculer PnL
        entry_price = pos["avg_price"]
        pnl = (price - entry_price) * amount

        # Mettre à jour la position
        pos["amount"] -= amount
        if pos["amount"] <= 0:
            del self._paper_positions[symbol]
        else:
            pos["cost_basis"] = pos["amount"] * pos["avg_price"]

        return pnl

    async def get_paper_summary(self) -> Dict[str, Any]:
        """Résumé complet du paper trading"""
        cfg = get_config()
        current_balance = self._paper_balance.get(cfg.base_currency, 0.0)

        # Calculer la valeur des positions
        positions_value = 0.0
        positions_detail = []

        for sym, pos in self._paper_positions.items():
            try:
                current_price = await self.get_market_price(sym)
                value = pos["amount"] * current_price
                unrealized_pnl = (current_price - pos["avg_price"]) * pos["amount"]

                positions_value += value
                positions_detail.append({
                    "symbol": sym,
                    "amount": pos["amount"],
                    "entry_price": pos["avg_price"],
                    "current_price": current_price,
                    "value_usdt": value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": (unrealized_pnl / pos["cost_basis"]) * 100 if pos["cost_basis"] > 0 else 0,
                })
            except Exception:
                continue

        total_equity = current_balance + positions_value
        total_pnl = total_equity - self._initial_balance
        total_pnl_pct = (total_pnl / self._initial_balance) * 100

        # Stats des trades
        winning_trades = [t for t in self._trade_history if t.get("pnl", 0) and t["pnl"] > 0]
        losing_trades = [t for t in self._trade_history if t.get("pnl", 0) and t["pnl"] < 0]

        return {
            "initial_balance": self._initial_balance,
            "current_balance": current_balance,
            "positions_value": positions_value,
            "total_equity": total_equity,
            "total_pnl_usdt": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": positions_detail,
            "stats": {
                "total_trades": len(self._paper_orders),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(self._trade_history) * 100 if self._trade_history else 0,
                "total_fees_paid": sum(o.get("fee", {}).get("cost", 0) for o in self._paper_orders),
            },
        }
```

**Critères de validation**:
- [ ] Simulation de slippage réaliste
- [ ] Calcul des frais
- [ ] Tracking précis des positions
- [ ] PnL réalisé et non-réalisé
- [ ] Statistiques complètes

---

## T5.4 - Mode de Transition (Paper → Live)

### T5.4.1 - Implémenter la validation avant passage en live
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
"""
Validations avant le passage en mode live.
"""

from typing import Dict, List, Tuple


@dataclass
class LiveReadinessCheck:
    """Résultat d'un check de préparation"""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


class LiveReadinessValidator:
    """Valide que le bot est prêt pour le live trading"""

    MIN_PAPER_TRADES = 50
    MIN_WIN_RATE = 0.4  # 40%
    MIN_PAPER_HOURS = 24
    MAX_DRAWDOWN_PCT = 10.0

    def __init__(self, paper_summary: Dict) -> None:
        self.summary = paper_summary
        self.checks: List[LiveReadinessCheck] = []

    def run_all_checks(self) -> Tuple[bool, List[LiveReadinessCheck]]:
        """Exécute tous les checks"""
        self.checks = []

        self._check_trade_count()
        self._check_win_rate()
        self._check_profitability()
        self._check_drawdown()
        self._check_api_config()

        all_passed = all(c.passed for c in self.checks if c.severity == "error")
        return all_passed, self.checks

    def _check_trade_count(self) -> None:
        """Vérifie le nombre de trades paper"""
        count = self.summary.get("stats", {}).get("total_trades", 0)
        passed = count >= self.MIN_PAPER_TRADES

        self.checks.append(LiveReadinessCheck(
            name="Paper trade count",
            passed=passed,
            message=f"{count} trades (min: {self.MIN_PAPER_TRADES})",
            severity="error" if not passed else "info",
        ))

    def _check_win_rate(self) -> None:
        """Vérifie le win rate"""
        win_rate = self.summary.get("stats", {}).get("win_rate", 0) / 100
        passed = win_rate >= self.MIN_WIN_RATE

        self.checks.append(LiveReadinessCheck(
            name="Win rate",
            passed=passed,
            message=f"{win_rate:.1%} (min: {self.MIN_WIN_RATE:.0%})",
            severity="warning" if not passed else "info",
        ))

    def _check_profitability(self) -> None:
        """Vérifie la profitabilité"""
        pnl_pct = self.summary.get("total_pnl_pct", 0)
        passed = pnl_pct > 0

        self.checks.append(LiveReadinessCheck(
            name="Profitability",
            passed=passed,
            message=f"{pnl_pct:+.2f}% PnL",
            severity="error" if not passed else "info",
        ))

    def _check_drawdown(self) -> None:
        """Vérifie le drawdown max"""
        # Nécessite tracking du drawdown dans paper summary
        drawdown = self.summary.get("max_drawdown_pct", 0)
        passed = drawdown < self.MAX_DRAWDOWN_PCT

        self.checks.append(LiveReadinessCheck(
            name="Max drawdown",
            passed=passed,
            message=f"{drawdown:.1f}% (max: {self.MAX_DRAWDOWN_PCT}%)",
            severity="warning" if not passed else "info",
        ))

    def _check_api_config(self) -> None:
        """Vérifie la configuration API"""
        from src.config import get_config
        cfg = get_config()

        has_key = bool(cfg.exchange_api_key)
        has_secret = bool(cfg.exchange_api_secret)
        passed = has_key and has_secret

        self.checks.append(LiveReadinessCheck(
            name="API configuration",
            passed=passed,
            message="API keys configured" if passed else "Missing API keys",
            severity="error" if not passed else "info",
        ))

    def get_summary(self) -> str:
        """Retourne un résumé textuel"""
        lines = ["=== Live Readiness Check ===\n"]

        for check in self.checks:
            status = "✅" if check.passed else ("⚠️" if check.severity == "warning" else "❌")
            lines.append(f"{status} {check.name}: {check.message}")

        all_passed = all(c.passed for c in self.checks if c.severity == "error")
        lines.append("")
        lines.append("✅ READY FOR LIVE" if all_passed else "❌ NOT READY - Fix errors first")

        return "\n".join(lines)
```

**Critères de validation**:
- [ ] Checks automatiques avant live
- [ ] Rapport clair
- [ ] Blocage si critères non remplis
- [ ] Warnings pour améliorations suggérées

---

## T5.5 - Intégration Complète

### T5.5.1 - Intégrer le RiskManager dans la boucle principale
**Priorité**: CRITIQUE
**Estimation**: Simple

Modifier `src/bot/loop.py` :

```python
# Dans TradingBotLoop.__init__
from src.tools.risk import get_risk_manager

def __init__(self, ...) -> None:
    # ... existing code ...
    self.risk_manager = get_risk_manager()

# Dans _act, avant chaque ordre
async def _act(self, snapshot: Dict, actions: List[Dict]) -> None:
    if not actions:
        return

    portfolio = snapshot.get("portfolio", {})
    balance = portfolio.get("balance_usdt", 0)
    positions = {p["symbol"]: p for p in portfolio.get("positions", [])}

    for action in actions:
        atype = action.get("type", "").upper()
        symbol = action.get("symbol")

        if not symbol:
            continue

        if atype == "OPEN":
            side = action.get("side", "buy").lower()
            size_pct = float(action.get("size_pct_equity", 0.02))

            # Calculer la taille sûre
            amount_usd = self.risk_manager.calculate_safe_order_size(
                balance, size_pct
            )

            # Vérifier la liquidité
            liquidity = await self.exchange_client.estimate_liquidity(symbol)

            # Risk check complet
            ok, reason = self.risk_manager.check_order(
                amount_usd=amount_usd,
                balance_usd=balance,
                symbol=symbol,
                side=side,
                current_positions=positions,
                liquidity_info=liquidity,
            )

            if not ok:
                self.memory.log_info(
                    f"Order rejected by risk manager: {reason}",
                    context={"action": action, "amount_usd": amount_usd},
                )
                if self.renderer:
                    self.renderer.add_action("REJECTED", symbol, reason[:40])
                continue

            # Exécuter l'ordre
            try:
                price = await self.exchange_client.get_market_price(symbol)
                amount = amount_usd / price

                order = await self.exchange_client.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                )

                self.memory.log_trade_open(order, snapshot, action)

                if self.renderer:
                    self.renderer.add_action(
                        "BUY" if side == "buy" else "SELL",
                        symbol,
                        f"${amount_usd:.2f} @ ${price:.2f}",
                    )

            except Exception as e:
                self.memory.log_error(f"Order execution failed: {e}")

        elif atype == "CLOSE":
            # ... existing close logic with renderer update ...
            pass

# À la fin de chaque loop, mettre à jour les stats risk
async def run_forever(self, interval_seconds: int = 60) -> None:
    # ... in the loop ...

    # Update risk manager avec l'equity actuelle
    portfolio = snapshot.get("portfolio", {})
    self.risk_manager.update_equity(portfolio.get("total_equity_usdt", 0))

    # Afficher les stats de session
    session_stats = self.risk_manager.get_session_stats()
    self.memory.log_info("Session stats", context=session_stats)
```

**Critères de validation**:
- [ ] Risk check avant chaque ordre
- [ ] Logging des rejets
- [ ] Update des stats de session
- [ ] UI mise à jour

---

## Checklist Finale Phase 5

- [ ] RiskManager avec toutes les règles
- [ ] Limites hard-coded incontournables
- [ ] Limites journalières avec halt
- [ ] Système d'alertes
- [ ] Paper trading avec slippage/fees
- [ ] Stats complètes paper trading
- [ ] Validation pre-live
- [ ] Intégration complète dans loop.py
- [ ] Tests unitaires risk manager
- [ ] Documentation des limites

---

## Notes Techniques

### Sécurité Critique
- Les limites max_order_usd et max_equity_pct sont des HARD LIMITS
- JAMAIS les contourner, même si le LLM demande plus
- Logging de TOUS les rejets pour audit

### Testnet vs Live
- Toujours démarrer sur testnet exchange
- Puis paper trading avec prix réels
- Enfin live avec micro-montants

### Monitoring
- Dashboard des stats de session
- Alertes en temps réel
- Export des logs pour analyse

---

## T5.6 - Architecture Modulaire Risk Management

> **Objectif**: Assurer que le RiskManager implémente l'interface `BaseRiskManager`.
> Permet de définir des règles de risque custom ou de swapper l'implémentation.

### T5.6.1 - Implémenter DefaultRiskManager (BaseRiskManager)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Modifier `src/tools/risk.py` pour implémenter l'interface :

```python
"""
Implémentation par défaut de BaseRiskManager.
Modulaire et configurable via règles extensibles.
"""

from typing import Any, Dict, List, Optional
from src.interfaces import BaseRiskManager, OrderRequest, RiskCheckResult
from src.config import get_config


class DefaultRiskManager(BaseRiskManager):
    """
    Implémentation concrète de BaseRiskManager.

    Fonctionnalités:
    - Règles configurables via RiskLimits
    - Rules extensibles via le pattern Chain of Responsibility
    - Stats de session persistantes
    """

    def __init__(
        self,
        config: Optional["Config"] = None,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        self._config = config or get_config()
        self.limits = limits or RiskLimits.from_config()
        self.session = TradingSession()
        self._rules: List["RiskRule"] = self._build_default_rules()

    def _build_default_rules(self) -> List["RiskRule"]:
        """Construit la chaîne de règles par défaut"""
        return [
            MaxOrderSizeRule(self.limits),
            MaxEquityPercentRule(self.limits),
            DailyLossLimitRule(self.limits, self.session),
            MaxPositionsRule(self.limits),
            SpreadLimitRule(self.limits),
            RateLimitRule(),
        ]

    def add_rule(self, rule: "RiskRule") -> None:
        """Ajoute une règle custom"""
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        """Retire une règle par son nom"""
        self._rules = [r for r in self._rules if r.name != rule_name]

    def check_order(
        self,
        order: OrderRequest,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> RiskCheckResult:
        """
        Implémente BaseRiskManager.check_order
        Exécute toutes les règles en chaîne.
        """
        # Vérifier la session
        if self.session.is_halted:
            return RiskCheckResult(
                approved=False,
                reason=f"Trading halted: {self.session.halt_reason}",
            )

        # Contexte pour les règles
        context = RiskContext(
            order=order,
            portfolio=portfolio_state,
            market=market_state,
            limits=self.limits,
            session=self.session,
        )

        # Exécuter chaque règle
        warnings = []
        for rule in self._rules:
            result = rule.check(context)
            if not result.approved:
                return result
            if result.warnings:
                warnings.extend(result.warnings)

        return RiskCheckResult(
            approved=True,
            adjusted_amount=order.amount,
            warnings=warnings,
        )

    def get_constraints(self) -> Dict[str, Any]:
        """Implémente BaseRiskManager.get_constraints"""
        return {
            "max_order_usd": self.limits.max_order_usd,
            "max_equity_pct": self.limits.max_equity_pct,
            "max_daily_loss_usd": self.limits.max_daily_loss_usd,
            "max_positions": self.limits.max_open_positions,
            "max_spread_pct": self.limits.max_spread_pct,
            "active_rules": [r.name for r in self._rules],
            "session_halted": self.session.is_halted,
        }

    def update_daily_stats(self, pnl: float) -> None:
        """Implémente BaseRiskManager.update_daily_stats"""
        self.session.trades_count += 1
        self.session.total_pnl_usd += pnl

        # Check daily loss limit
        if self.session.total_pnl_usd < -self.limits.max_daily_loss_usd:
            self.session.is_halted = True
            self.session.halt_reason = f"Daily loss limit: ${-self.session.total_pnl_usd:.2f}"

    def should_halt(self) -> bool:
        """Implémente BaseRiskManager.should_halt"""
        return self.session.is_halted
```

**Critères de validation**:
- [ ] Implémente toutes les méthodes de BaseRiskManager
- [ ] Règles extensibles via add_rule/remove_rule
- [ ] Contexte riche pour les règles
- [ ] Session tracking

---

### T5.6.2 - Créer le pattern RiskRule (Chain of Responsibility)
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
"""
Pattern Chain of Responsibility pour les règles de risque.
Chaque règle peut approuver, rejeter ou ajouter des warnings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from src.interfaces import OrderRequest, RiskCheckResult


@dataclass
class RiskContext:
    """Contexte passé à chaque règle"""
    order: OrderRequest
    portfolio: Dict[str, Any]
    market: Dict[str, Any]
    limits: "RiskLimits"
    session: "TradingSession"


class RiskRule(ABC):
    """Classe de base pour une règle de risque"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la règle pour logging/config"""
        ...

    @abstractmethod
    def check(self, context: RiskContext) -> RiskCheckResult:
        """
        Vérifie la règle.
        Return RiskCheckResult(approved=True) pour continuer,
        ou RiskCheckResult(approved=False, reason=...) pour rejeter.
        """
        ...


# === RÈGLES STANDARD ===

class MaxOrderSizeRule(RiskRule):
    """Limite la taille max d'un ordre en USD"""

    def __init__(self, limits: "RiskLimits") -> None:
        self._limits = limits

    @property
    def name(self) -> str:
        return "max_order_size"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        if ctx.order.amount > self._limits.max_order_usd:
            return RiskCheckResult(
                approved=False,
                reason=f"Order too large: ${ctx.order.amount:.2f} > ${self._limits.max_order_usd}",
            )
        return RiskCheckResult(approved=True)


class MaxEquityPercentRule(RiskRule):
    """Limite le % max de l'equity par ordre"""

    def __init__(self, limits: "RiskLimits") -> None:
        self._limits = limits

    @property
    def name(self) -> str:
        return "max_equity_percent"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        balance = ctx.portfolio.get("balance_usdt", 0)
        if balance <= 0:
            return RiskCheckResult(approved=False, reason="Zero balance")

        pct = ctx.order.amount / balance
        if pct > self._limits.max_equity_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Exceeds equity limit: {pct:.1%} > {self._limits.max_equity_pct:.1%}",
            )
        return RiskCheckResult(approved=True)


class DailyLossLimitRule(RiskRule):
    """Arrête le trading si perte journalière dépassée"""

    def __init__(self, limits: "RiskLimits", session: "TradingSession") -> None:
        self._limits = limits
        self._session = session

    @property
    def name(self) -> str:
        return "daily_loss_limit"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        if ctx.session.total_pnl_usd < -self._limits.max_daily_loss_usd:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit exceeded: ${-ctx.session.total_pnl_usd:.2f}",
            )
        return RiskCheckResult(approved=True)


class MaxPositionsRule(RiskRule):
    """Limite le nombre de positions ouvertes"""

    def __init__(self, limits: "RiskLimits") -> None:
        self._limits = limits

    @property
    def name(self) -> str:
        return "max_positions"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        positions = ctx.portfolio.get("positions", [])
        if len(positions) >= self._limits.max_open_positions:
            # Vérifie si c'est une nouvelle position
            existing = {p["symbol"] for p in positions}
            if ctx.order.symbol not in existing:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Max positions reached: {self._limits.max_open_positions}",
                )
        return RiskCheckResult(approved=True)


class SpreadLimitRule(RiskRule):
    """Rejette si le spread est trop élevé"""

    def __init__(self, limits: "RiskLimits") -> None:
        self._limits = limits

    @property
    def name(self) -> str:
        return "spread_limit"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        spread = ctx.market.get("spread_pct", 0)
        if spread > self._limits.max_spread_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Spread too high: {spread:.2f}% > {self._limits.max_spread_pct}%",
            )
        return RiskCheckResult(approved=True)


class RateLimitRule(RiskRule):
    """Anti-spam: min 5s entre ordres sur même symbole"""

    def __init__(self) -> None:
        self._last_order: Dict[str, float] = {}
        self._min_interval = 5.0  # seconds

    @property
    def name(self) -> str:
        return "rate_limit"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        import time
        now = time.time()
        symbol = ctx.order.symbol
        last = self._last_order.get(symbol, 0)

        if (now - last) < self._min_interval:
            return RiskCheckResult(
                approved=False,
                reason=f"Rate limit: wait {self._min_interval}s between orders",
            )

        self._last_order[symbol] = now
        return RiskCheckResult(approved=True)
```

**Critères de validation**:
- [ ] Pattern Chain of Responsibility
- [ ] Règles indépendantes et testables
- [ ] Facile d'ajouter une règle custom
- [ ] Contexte riche pour flexibilité

---

### T5.6.3 - Créer des règles custom (exemple de modularité)
**Priorité**: MOYENNE
**Estimation**: Simple

Exemple montrant comment ajouter une règle custom :

```python
"""
Règles de risque custom - exemple de modularité.
Il suffit d'hériter de RiskRule et d'implémenter check().
"""

class VolatilityLimitRule(RiskRule):
    """
    Règle custom: rejette si la volatilité est trop élevée.
    Démontre l'extensibilité du système.
    """

    def __init__(self, max_volatility: float = 0.10) -> None:
        self._max_vol = max_volatility

    @property
    def name(self) -> str:
        return "volatility_limit"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        vol = ctx.market.get("volatility", 0)
        if vol > self._max_vol:
            return RiskCheckResult(
                approved=False,
                reason=f"Volatility too high: {vol:.1%} > {self._max_vol:.1%}",
            )
        return RiskCheckResult(approved=True)


class NarrativeStrengthRule(RiskRule):
    """
    Règle custom: n'achète que si le narratif est fort.
    Spécifique à la stratégie SocialFi/Trends.
    """

    def __init__(self, min_strength: float = 0.5) -> None:
        self._min_strength = min_strength

    @property
    def name(self) -> str:
        return "narrative_strength"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        # Uniquement pour les achats
        if ctx.order.side != "buy":
            return RiskCheckResult(approved=True)

        strength = ctx.market.get("narrative_strength", 1.0)
        if strength < self._min_strength:
            return RiskCheckResult(
                approved=False,
                reason=f"Narrative too weak: {strength:.1%} < {self._min_strength:.1%}",
                warnings=["Consider waiting for stronger narrative signal"],
            )
        return RiskCheckResult(approved=True)


class MaxDrawdownRule(RiskRule):
    """Arrête le trading si le drawdown dépasse un seuil"""

    def __init__(self, max_drawdown_pct: float = 0.10) -> None:
        self._max_dd = max_drawdown_pct

    @property
    def name(self) -> str:
        return "max_drawdown"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        peak = ctx.session.peak_equity_usd
        current = ctx.portfolio.get("total_equity_usdt", peak)

        if peak > 0:
            drawdown = (peak - current) / peak
            if drawdown > self._max_dd:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Max drawdown exceeded: {drawdown:.1%} > {self._max_dd:.1%}",
                )

        return RiskCheckResult(approved=True)


# === USAGE ===

def create_aggressive_risk_manager() -> DefaultRiskManager:
    """Factory pour un RiskManager avec règles agressives"""
    limits = RiskLimits(
        max_order_usd=50.0,  # Plus élevé
        max_equity_pct=0.10,  # 10% max
        max_daily_loss_usd=100.0,
    )
    rm = DefaultRiskManager(limits=limits)
    # Pas de règle de volatilité
    rm.remove_rule("volatility_limit")
    return rm


def create_conservative_risk_manager() -> DefaultRiskManager:
    """Factory pour un RiskManager conservateur"""
    limits = RiskLimits(
        max_order_usd=10.0,  # Plus bas
        max_equity_pct=0.02,  # 2% max
        max_daily_loss_usd=20.0,
    )
    rm = DefaultRiskManager(limits=limits)
    # Ajouter des règles supplémentaires
    rm.add_rule(VolatilityLimitRule(max_volatility=0.05))
    rm.add_rule(MaxDrawdownRule(max_drawdown_pct=0.05))
    return rm


def create_socialfi_risk_manager() -> DefaultRiskManager:
    """Factory pour la stratégie SocialFi"""
    rm = DefaultRiskManager()
    rm.add_rule(NarrativeStrengthRule(min_strength=0.6))
    return rm
```

**Comment ajouter une règle custom:**
1. Créer une classe qui hérite de `RiskRule`
2. Implémenter la propriété `name`
3. Implémenter la méthode `check(context) -> RiskCheckResult`
4. Ajouter avec `risk_manager.add_rule(MyRule())`

**Critères de validation**:
- [ ] Règles custom fonctionnelles
- [ ] Pas de modification du code principal
- [ ] Factories pour différentes stratégies
- [ ] Documentation claire

---

### T5.6.4 - Configuration des règles via environnement
**Priorité**: BASSE
**Estimation**: Simple

```python
"""
Configuration des règles de risque via .env ou config.
"""

import os
from typing import List, Optional


def load_risk_rules_from_env() -> List[RiskRule]:
    """
    Charge les règles depuis l'environnement.

    Variables:
    - RISK_DISABLED_RULES: "rule1,rule2" - règles à désactiver
    - RISK_MAX_ORDER_USD: override du max order
    - RISK_MAX_VOLATILITY: ajoute une règle de volatilité
    """
    limits = RiskLimits()

    # Override du max order
    max_order = os.getenv("RISK_MAX_ORDER_USD")
    if max_order:
        limits.max_order_usd = float(max_order)

    # Règles de base
    rules = [
        MaxOrderSizeRule(limits),
        MaxEquityPercentRule(limits),
        DailyLossLimitRule(limits),
        MaxPositionsRule(limits),
        SpreadLimitRule(limits),
        RateLimitRule(),
    ]

    # Règle de volatilité optionnelle
    max_vol = os.getenv("RISK_MAX_VOLATILITY")
    if max_vol:
        rules.append(VolatilityLimitRule(float(max_vol)))

    # Règles désactivées
    disabled = os.getenv("RISK_DISABLED_RULES", "").split(",")
    rules = [r for r in rules if r.name not in disabled]

    return rules


# .env.example additions
"""
# === Risk Management ===
RISK_MAX_ORDER_USD=20.0
RISK_MAX_EQUITY_PCT=0.05
RISK_MAX_DAILY_LOSS_USD=50.0
RISK_MAX_VOLATILITY=0.10
RISK_DISABLED_RULES=                # e.g., "rate_limit,spread_limit"
"""
```

**Critères de validation**:
- [ ] Config via environnement
- [ ] Override des limites
- [ ] Règles activables/désactivables
- [ ] Documentation dans .env.example
