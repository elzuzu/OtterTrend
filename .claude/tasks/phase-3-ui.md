# Phase 3 - UI Rich / Renderer CLI

> **Objectif**: Implémenter l'interface terminal avec la librairie Rich pour visualiser en temps réel les pensées du LLM, les actions, et le statut du bot.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (structure de base)
- Phase 1 partielle (portfolio state)
- Phase 2 partielle (trends snapshot)

---

## T3.1 - Architecture du Renderer

### T3.1.1 - Concevoir le layout principal
**Priorité**: HAUTE
**Estimation**: Simple

**Layout cible** :
```
┌──────────────────────────────────────────────────────────────────────────┐
│                           OtterTrend v1.0                                │
│                    Mode: PAPER | Exchange: MEXC                          │
├────────────────────────────────────┬─────────────────────────────────────┤
│         🧠 Thoughts (LLM)          │          📊 Portfolio               │
│                                    │  Balance: $1,234.56 USDT            │
│  Analyzing BTC/USDT trends...      │  Equity:  $1,456.78                  │
│  Google Trends shows rising        │  PnL:     +$56.78 (+4.5%)           │
│  interest in "bitcoin etf"         │                                     │
│  News sentiment: positive (+0.6)   │  Positions:                         │
│                                    │  • BTC/USDT: 0.001 @ $42,000        │
│  Decision: Open small position     │  • ETH/USDT: 0.01 @ $2,200          │
│  on BTC/USDT (2% equity)...        │                                     │
│                                    │                                     │
├────────────────────────────────────┼─────────────────────────────────────┤
│         ⚡ Recent Actions          │          📈 Market                  │
│                                    │                                     │
│  [12:34:56] BUY BTC/USDT          │  BTC/USDT: $43,250 (+2.3%)          │
│            0.001 @ $43,200        │  ETH/USDT: $2,280  (+1.8%)          │
│            Status: FILLED          │  SOL/USDT: $98.50  (+5.2%)          │
│                                    │                                     │
│  [12:30:00] CLOSE ETH/USDT        │  Trends: 🔥 AI crypto               │
│            PnL: +$12.50 (+1.2%)   │          📈 socialfi                │
│                                    │          📊 memecoin               │
│                                    │                                     │
├────────────────────────────────────┴─────────────────────────────────────┤
│  Status: Running | Loop: #42 | Last update: 12:35:00 | Next: 60s        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Critères de validation**:
- [ ] Layout défini avec 4 zones principales
- [ ] Header avec infos générales
- [ ] Footer avec statut
- [ ] Responsive (s'adapte à la taille du terminal)

---

### T3.1.2 - Créer la structure de base du Renderer
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `src/ui/renderer.py` :

```python
"""
Renderer Rich pour OtterTrend.
Affiche en temps réel les pensées du LLM, les actions et le portfolio.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.spinner import Spinner


@dataclass
class RendererState:
    """État interne du renderer"""
    thought_buffer: str = ""
    recent_actions: List[Dict] = field(default_factory=list)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)
    trends_data: Dict[str, Any] = field(default_factory=dict)
    loop_count: int = 0
    last_update: Optional[datetime] = None
    next_update_in: int = 0
    status: str = "Initializing"
    mode: str = "PAPER"
    exchange: str = "MEXC"


class Renderer:
    """
    Gestionnaire d'affichage Rich pour le terminal.
    Utilise Rich Live pour les mises à jour en temps réel.
    """

    MAX_THOUGHTS_LENGTH = 2000
    MAX_ACTIONS = 10

    def __init__(self) -> None:
        self.console = Console()
        self.state = RendererState()
        self._live: Optional[Live] = None

    def _create_layout(self) -> Layout:
        """Crée la structure du layout"""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].split(
            Layout(name="thoughts", ratio=2),
            Layout(name="actions", ratio=1),
        )

        layout["right"].split(
            Layout(name="portfolio", ratio=1),
            Layout(name="market", ratio=1),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render le header"""
        title = Text()
        title.append("🦦 OtterTrend", style="bold cyan")
        title.append(" v1.0 | ", style="dim")
        title.append(f"Mode: ", style="dim")
        title.append(
            self.state.mode,
            style="green" if self.state.mode == "PAPER" else "red bold"
        )
        title.append(f" | Exchange: ", style="dim")
        title.append(self.state.exchange, style="yellow")

        return Panel(title, style="blue")

    def _render_thoughts(self) -> Panel:
        """Render les pensées du LLM"""
        content = Text(self.state.thought_buffer[-self.MAX_THOUGHTS_LENGTH:])
        return Panel(
            content,
            title="🧠 Thoughts (LLM)",
            border_style="cyan",
        )

    def _render_actions(self) -> Panel:
        """Render les actions récentes"""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Action", width=8)
        table.add_column("Symbol", width=12)
        table.add_column("Details", width=20)

        for action in self.state.recent_actions[-self.MAX_ACTIONS:]:
            time_str = action.get("time", "")[:8]
            action_type = action.get("type", "")
            symbol = action.get("symbol", "")
            details = action.get("details", "")

            # Couleur selon le type
            if action_type in ("BUY", "OPEN"):
                style = "green"
            elif action_type in ("SELL", "CLOSE"):
                style = "red"
            else:
                style = "yellow"

            table.add_row(
                time_str,
                Text(action_type, style=style),
                symbol,
                details,
            )

        return Panel(table, title="⚡ Recent Actions", border_style="magenta")

    def _render_portfolio(self) -> Panel:
        """Render le portfolio"""
        p = self.state.portfolio
        content = Text()

        balance = p.get("balance_usdt", 0)
        equity = p.get("total_equity_usdt", 0)
        pnl = p.get("pnl_usdt", 0)
        pnl_pct = p.get("pnl_pct", 0)

        content.append(f"Balance: ", style="dim")
        content.append(f"${balance:,.2f} USDT\n", style="bold")

        content.append(f"Equity:  ", style="dim")
        content.append(f"${equity:,.2f}\n", style="bold")

        content.append(f"PnL:     ", style="dim")
        pnl_style = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""
        content.append(
            f"{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)\n",
            style=pnl_style
        )

        content.append("\nPositions:\n", style="bold")

        positions = p.get("positions", [])
        if positions:
            for pos in positions[:5]:
                sym = pos.get("symbol", "")
                amt = pos.get("amount", 0)
                price = pos.get("price", 0)
                content.append(f"• {sym}: ", style="dim")
                content.append(f"{amt:.6f} @ ${price:,.2f}\n")
        else:
            content.append("  No open positions\n", style="dim italic")

        return Panel(content, title="📊 Portfolio", border_style="green")

    def _render_market(self) -> Panel:
        """Render les données marché et trends"""
        content = Text()

        # Tickers
        tickers = self.state.market_data.get("tickers", {})
        for sym, data in list(tickers.items())[:5]:
            price = data.get("last", 0)
            change = data.get("change_24h_pct", 0)
            change_style = "green" if change >= 0 else "red"
            change_sign = "+" if change >= 0 else ""

            content.append(f"{sym}: ", style="dim")
            content.append(f"${price:,.2f} ", style="bold")
            content.append(f"({change_sign}{change:.1f}%)\n", style=change_style)

        # Trends
        content.append("\nTrends:\n", style="bold")
        trends = self.state.trends_data.get("trending_narratives", [])
        if trends:
            for trend in trends[:3]:
                kw = trend.get("keyword", "")
                score = trend.get("current_score", 0)
                emoji = "🔥" if score > 80 else "📈" if score > 50 else "📊"
                content.append(f"{emoji} {kw}\n")
        else:
            content.append("  Loading trends...\n", style="dim italic")

        return Panel(content, title="📈 Market", border_style="yellow")

    def _render_footer(self) -> Panel:
        """Render le footer avec statut"""
        status = Text()
        status.append("Status: ", style="dim")
        status.append(self.state.status, style="bold green")
        status.append(f" | Loop: #{self.state.loop_count}", style="dim")

        if self.state.last_update:
            status.append(
                f" | Last: {self.state.last_update.strftime('%H:%M:%S')}",
                style="dim"
            )

        status.append(f" | Next: {self.state.next_update_in}s", style="dim")

        return Panel(status, style="blue")

    def render(self) -> Layout:
        """Génère le layout complet"""
        layout = self._create_layout()
        layout["header"].update(self._render_header())
        layout["thoughts"].update(self._render_thoughts())
        layout["actions"].update(self._render_actions())
        layout["portfolio"].update(self._render_portfolio())
        layout["market"].update(self._render_market())
        layout["footer"].update(self._render_footer())
        return layout
```

**Critères de validation**:
- [ ] Layout 4 zones fonctionnel
- [ ] Tous les panels rendus correctement
- [ ] Styles et couleurs appropriés
- [ ] Gestion des données manquantes

---

## T3.2 - Méthodes de Mise à Jour

### T3.2.1 - Implémenter les méthodes de mise à jour d'état
**Priorité**: CRITIQUE
**Estimation**: Simple

Ajouter à la classe `Renderer` :

```python
    # === Méthodes de mise à jour ===

    def write_thought(self, token: str) -> None:
        """Ajoute un token au buffer de pensées (streaming)"""
        self.state.thought_buffer += token
        # Limiter la taille
        if len(self.state.thought_buffer) > self.MAX_THOUGHTS_LENGTH * 2:
            self.state.thought_buffer = self.state.thought_buffer[-self.MAX_THOUGHTS_LENGTH:]

    def clear_thoughts(self) -> None:
        """Efface le buffer de pensées"""
        self.state.thought_buffer = ""

    def set_thoughts(self, text: str) -> None:
        """Remplace le buffer de pensées"""
        self.state.thought_buffer = text

    def add_action(
        self,
        action_type: str,
        symbol: str,
        details: str = "",
    ) -> None:
        """Ajoute une action à la liste"""
        self.state.recent_actions.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": action_type.upper(),
            "symbol": symbol,
            "details": details,
        })
        # Limiter le nombre d'actions
        if len(self.state.recent_actions) > self.MAX_ACTIONS * 2:
            self.state.recent_actions = self.state.recent_actions[-self.MAX_ACTIONS:]

    def update_portfolio(self, portfolio: Dict[str, Any]) -> None:
        """Met à jour les données du portfolio"""
        self.state.portfolio = portfolio

    def update_market(self, market_data: Dict[str, Any]) -> None:
        """Met à jour les données marché"""
        self.state.market_data = market_data

    def update_trends(self, trends_data: Dict[str, Any]) -> None:
        """Met à jour les données de tendances"""
        self.state.trends_data = trends_data

    def update_status(
        self,
        status: str,
        loop_count: Optional[int] = None,
        next_update_in: Optional[int] = None,
    ) -> None:
        """Met à jour le statut"""
        self.state.status = status
        self.state.last_update = datetime.now()
        if loop_count is not None:
            self.state.loop_count = loop_count
        if next_update_in is not None:
            self.state.next_update_in = next_update_in

    def set_mode(self, mode: str, exchange: str) -> None:
        """Configure le mode et l'exchange"""
        self.state.mode = mode
        self.state.exchange = exchange
```

**Critères de validation**:
- [ ] Toutes les méthodes de mise à jour implémentées
- [ ] Limites de buffer respectées
- [ ] Timestamps corrects
- [ ] Thread-safe (pour usage async)

---

## T3.3 - Mode Live (Temps Réel)

### T3.3.1 - Implémenter le mode Live avec Rich
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
    # === Mode Live ===

    def start_live(self, refresh_rate: int = 4) -> None:
        """Démarre le mode Live (mise à jour automatique)"""
        if self._live is not None:
            return

        self._live = Live(
            self.render(),
            refresh_per_second=refresh_rate,
            console=self.console,
            screen=True,  # Utilise tout l'écran
        )
        self._live.start()

    def stop_live(self) -> None:
        """Arrête le mode Live"""
        if self._live:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        """Force un rafraîchissement de l'affichage"""
        if self._live:
            self._live.update(self.render())

    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_live()
        return False


# === Fonction utilitaire pour le mode simple ===

def print_status(
    console: Console,
    message: str,
    level: str = "INFO",
) -> None:
    """Affiche un message de statut simple"""
    styles = {
        "INFO": "cyan",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "DEBUG": "dim",
    }
    style = styles.get(level, "white")
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(f"[{timestamp}] [{level}] {message}", style=style)
```

**Critères de validation**:
- [ ] Mode Live fonctionnel
- [ ] Refresh automatique
- [ ] Context manager pour cleanup
- [ ] Pas de flicker

---

### T3.3.2 - Implémenter le mode simplifié (sans Live)
**Priorité**: MOYENNE
**Estimation**: Simple

Pour les cas où Live n'est pas souhaité (logs, debug) :

```python
class SimpleRenderer:
    """
    Version simplifiée du renderer pour le debug/logs.
    Pas de mise à jour en temps réel, juste des prints formatés.
    """

    def __init__(self) -> None:
        self.console = Console()

    def print_thought(self, thought: str) -> None:
        """Affiche une pensée du LLM"""
        self.console.print(Panel(thought, title="🧠 Thought", border_style="cyan"))

    def print_action(
        self,
        action_type: str,
        symbol: str,
        details: str = "",
    ) -> None:
        """Affiche une action"""
        style = "green" if action_type in ("BUY", "OPEN") else "red"
        self.console.print(
            f"⚡ [{action_type}] {symbol} - {details}",
            style=style
        )

    def print_portfolio(self, portfolio: Dict[str, Any]) -> None:
        """Affiche le portfolio"""
        table = Table(title="📊 Portfolio")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Balance", f"${portfolio.get('balance_usdt', 0):,.2f}")
        table.add_row("Equity", f"${portfolio.get('total_equity_usdt', 0):,.2f}")

        pnl = portfolio.get("pnl_usdt", 0)
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row("PnL", f"[{pnl_style}]${pnl:,.2f}[/]")

        self.console.print(table)

    def print_error(self, message: str) -> None:
        """Affiche une erreur"""
        self.console.print(f"❌ ERROR: {message}", style="bold red")

    def print_info(self, message: str) -> None:
        """Affiche une info"""
        self.console.print(f"ℹ️  {message}", style="cyan")

    def print_success(self, message: str) -> None:
        """Affiche un succès"""
        self.console.print(f"✅ {message}", style="green")
```

**Critères de validation**:
- [ ] Rendu simple et clair
- [ ] Utilisable pour debug
- [ ] Pas de dépendance sur Live

---

## T3.4 - Intégration avec la Boucle Principale

### T3.4.1 - Intégrer le Renderer dans TradingBotLoop
**Priorité**: HAUTE
**Estimation**: Moyenne

Modifier `src/bot/loop.py` pour utiliser le renderer :

```python
# Dans TradingBotLoop.__init__
def __init__(
    self,
    groq_client: GroqAdapter,
    renderer: Optional[Renderer] = None,
) -> None:
    self.groq = groq_client
    self.memory = BotMemory("bot_data.db")
    self.renderer = renderer
    self._loop_count = 0

# Dans run_forever
async def run_forever(self, interval_seconds: int = 60) -> None:
    if self.renderer:
        self.renderer.set_mode(
            "PAPER" if get_config().paper_trading else "LIVE",
            get_config().exchange_id.upper(),
        )

    while True:
        self._loop_count += 1
        loop_started_at = datetime.utcnow()

        if self.renderer:
            self.renderer.update_status(
                "Observing",
                loop_count=self._loop_count,
                next_update_in=interval_seconds,
            )
            self.renderer.clear_thoughts()

        try:
            # === OBSERVE ===
            if self.renderer:
                self.renderer.write_thought("📊 Fetching market data...\n")

            snapshot = await self._observe()

            if self.renderer:
                self.renderer.update_portfolio(snapshot.get("portfolio", {}))
                self.renderer.update_market(snapshot.get("market", {}))
                self.renderer.update_trends(snapshot.get("trends", {}))
                self.renderer.update_status("Thinking")

            # === THINK ===
            actions = await self._think(snapshot)

            if self.renderer:
                self.renderer.update_status("Acting")

            # === ACT ===
            await self._act(snapshot, actions)

            if self.renderer:
                self.renderer.update_status("Waiting")

        except Exception as exc:
            if self.renderer:
                self.renderer.update_status(f"Error: {str(exc)[:30]}")
            self.memory.log_error(f"Loop error: {exc}", context={})

        finally:
            elapsed = (datetime.utcnow() - loop_started_at).total_seconds()
            sleep_for = max(0, interval_seconds - elapsed)

            if self.renderer:
                # Countdown
                for remaining in range(int(sleep_for), 0, -1):
                    self.renderer.update_status(
                        "Waiting",
                        next_update_in=remaining,
                    )
                    self.renderer.refresh()
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(sleep_for)

# Dans _think, streamer les tokens vers le renderer
async def _think(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ... code existant ...

    async for event in self._stream_llm(messages):
        if event["type"] == "token":
            token = event["content"]
            buffered_text += token
            if self.renderer:
                self.renderer.write_thought(token)
                self.renderer.refresh()
        # ... reste du code ...
```

**Critères de validation**:
- [ ] Renderer optionnel (peut fonctionner sans)
- [ ] Mise à jour en temps réel pendant Think
- [ ] Countdown visible pendant le wait
- [ ] Statut clair à chaque étape

---

### T3.4.2 - Modifier main.py pour activer le renderer
**Priorité**: HAUTE
**Estimation**: Simple

```python
# main.py

async def main() -> int:
    load_dotenv()
    cfg = get_config()

    # ... validation ...

    groq_client = GroqAdapter(
        api_key=cfg.groq_api_key,
        model=cfg.llm_model,
        system_prompt=SYSTEM_PROMPT,
    )

    # Créer le renderer
    renderer = Renderer()

    bot_loop = TradingBotLoop(
        groq_client=groq_client,
        renderer=renderer,
    )

    # Démarrer avec le renderer en mode Live
    with renderer:
        try:
            await bot_loop.run_forever(interval_seconds=cfg.loop_interval_seconds)
        except KeyboardInterrupt:
            pass

    return 0
```

**Critères de validation**:
- [ ] Context manager pour cleanup automatique
- [ ] Ctrl+C fonctionne proprement
- [ ] Affichage correct au démarrage

---

## T3.5 - Composants Additionnels

### T3.5.1 - Spinner et messages de chargement
**Priorité**: BASSE
**Estimation**: Simple

```python
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn

def create_loading_spinner(message: str) -> Spinner:
    """Crée un spinner de chargement"""
    return Spinner("dots", text=message, style="cyan")

async def with_spinner(
    console: Console,
    message: str,
    coro,
):
    """Exécute une coroutine avec un spinner"""
    with console.status(message, spinner="dots"):
        return await coro
```

---

### T3.5.2 - Barre de progression pour les opérations longues
**Priorité**: BASSE
**Estimation**: Simple

```python
from rich.progress import Progress, TaskID

class ProgressTracker:
    """Tracker de progression pour opérations multi-étapes"""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._progress: Optional[Progress] = None

    def start(self, total_steps: int, description: str = "Processing") -> TaskID:
        self._progress = Progress(console=self.console)
        self._progress.start()
        return self._progress.add_task(description, total=total_steps)

    def advance(self, task_id: TaskID, advance: int = 1) -> None:
        if self._progress:
            self._progress.update(task_id, advance=advance)

    def stop(self) -> None:
        if self._progress:
            self._progress.stop()
            self._progress = None
```

---

## Checklist Finale Phase 3

- [ ] Layout 4 zones avec header/footer
- [ ] Panel Thoughts avec streaming
- [ ] Panel Actions avec historique
- [ ] Panel Portfolio avec PnL
- [ ] Panel Market avec tickers et trends
- [ ] Mode Live fonctionnel
- [ ] Mode Simple pour debug
- [ ] Intégration avec TradingBotLoop
- [ ] Couleurs et styles appropriés
- [ ] Responsive (petits terminaux)
- [ ] Gestion propre de Ctrl+C

---

## Notes Techniques

### Performance Rich
- Limiter le refresh_per_second (4-10 max)
- Éviter les rendus trop fréquents dans les boucles
- Utiliser `refresh()` uniquement quand nécessaire

### Compatibilité Terminaux
- Tester sur différents terminaux (iTerm, Terminal, Windows Terminal)
- Fallback gracieux si Rich non supporté
- Mode simple pour SSH sans support couleurs

### Thread Safety
- Rich Console est thread-safe
- Utiliser des locks si mise à jour depuis plusieurs sources
- Préférer asyncio à threading
