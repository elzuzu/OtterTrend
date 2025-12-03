# Phase 0 - Setup & Architecture de Base

> **Objectif**: Mettre en place le repository, la structure de fichiers et les configurations de base pour le bot OtterTrend - fork conceptuel de gemini-cli avec Groq + MEXC.
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
1. **Frais quasi nuls** - Critical pour 10-20 trades/jour
2. **Listings agressifs** - Tokens SocialFi dispo des semaines avant OKX/Binance
3. **Scalping possible** - Avec 0% fees maker, petits mouvements rentables

**Note sécurité** : MEXC = plateforme de **transit et d'exécution**, pas de stockage long terme.

## Statut Global
- [ ] Phase complète

---

## T0.1 - Structure du Projet

### T0.1.1 - Créer l'arborescence de fichiers
**Priorité**: CRITIQUE
**Estimation**: Simple

Créer la structure suivante (alignée ChatGPT - Observer/Réfléchir/Agir) :
```
OtterTrend/
├── main.py                          # Point d'entrée (boucle autonome)
├── requirements.txt                 # Dépendances Python
├── .env.example                     # Template des variables d'environnement
├── .gitignore                       # Fichiers à ignorer
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration centralisée
│   ├── client/
│   │   ├── __init__.py
│   │   └── groq_adapter.py         # Adaptateur LLM Groq (remplace Gemini)
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── brain.py                # Policy LLM autonome
│   │   ├── memory.py               # Persistance SQLite3
│   │   └── loop.py                 # Orchestrateur Observe→Think→Act
│   ├── tools/
│   │   ├── __init__.py
│   │   │
│   │   ├── # === OBSERVER (données brutes & trends) ===
│   │   ├── market.py               # get_market_snapshot, get_orderbook (MEXC/CCXT)
│   │   ├── trends.py               # get_google_trends, get_trending_tokens
│   │   ├── social.py               # get_social_mentions, get_social_trending
│   │   ├── news.py                 # get_crypto_news, get_project_announcements
│   │   │
│   │   ├── # === RÉFLÉCHIR (mini-ML spécialisés) ===
│   │   ├── analytics.py            # ml_detect_regime, ml_forecast_volatility
│   │   ├── sentiment.py            # ml_score_sentiment, ml_narrative_strength
│   │   │
│   │   ├── # === AGIR (portfolio, risk, exécution) ===
│   │   ├── portfolio.py            # get_portfolio_state, risk_constraints
│   │   ├── risk.py                 # risk_check_order (garde-fous hard-coded)
│   │   ├── execution.py            # place_order, close_position (MEXC)
│   │   │
│   │   └── schemas.py              # Définitions JSON tools pour Groq
│   └── ui/
│       ├── __init__.py
│       └── renderer.py             # UI Rich (style gemini-cli)
└── tests/
    ├── __init__.py
    └── ...
```

**Critères de validation**:
- [ ] Tous les répertoires créés
- [ ] Fichiers `__init__.py` présents dans chaque module
- [ ] Structure importable (`from src.bot import loop`)

---

### T0.1.2 - Créer le fichier requirements.txt
**Priorité**: CRITIQUE
**Estimation**: Simple

```txt
# LLM
groq>=0.4.0

# Trading
ccxt>=4.0.0

# Trends & Data
pytrends>=4.9.0
requests>=2.31.0
numpy>=1.24.0

# UI
rich>=13.0.0

# Utilities
python-dotenv>=1.0.0
sqlite-utils>=3.35.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0
```

**Critères de validation**:
- [ ] Fichier créé à la racine
- [ ] `pip install -r requirements.txt` fonctionne sans erreur
- [ ] Toutes les versions compatibles Python 3.10+

---

### T0.1.3 - Créer le fichier .env.example
**Priorité**: HAUTE
**Estimation**: Simple

```env
# === LLM Provider (Groq Free Tier) ===
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=llama-3.3-70b-versatile   # Meilleur ratio reasoning/vitesse

# === Exchange MEXC (Frais bas + Listings rapides) ===
MEXC_API_KEY=your_api_key
MEXC_API_SECRET=your_api_secret
# Note: MEXC n'a PAS de passphrase (contrairement à OKX)

# === Trading Mode ===
PAPER_TRADING=true                  # true = simulation, false = réel
BASE_CURRENCY=USDT

# === Bot Settings ===
LOOP_INTERVAL_SECONDS=300           # Intervalle entre les cycles (5min recommandé)

# === Risk Limits (Hard-coded - ChatGPT spec) ===
MAX_ORDER_USD=20.0                  # Limite absolue par ordre
MAX_EQUITY_PCT=0.05                 # Max 5% du portefeuille par trade
MAX_DAILY_LOSS_USD=50.0             # HALT si perte > $50/jour
MAX_POSITIONS=5                     # Max positions simultanées

# === Narratifs & Trends (ChatGPT spec) ===
GOOGLE_TRENDS_KEYWORDS=socialfi,ai crypto,memecoin,airdrop,crypto
SOCIALFI_TOKENS=CYBER,DEGEN,LENS,ID

# === MEXC Spécifique ===
# Note: Frais 0% maker / 0.01% taker - idéal pour high-frequency trading
# Note: API plus stricte sur rate limits - délai recommandé entre appels
```

**Critères de validation**:
- [ ] Toutes les variables documentées
- [ ] Valeurs par défaut sécurisées (PAPER_TRADING=true)
- [ ] Fichier `.env` ajouté au `.gitignore`

---

### T0.1.4 - Créer le fichier .gitignore
**Priorité**: HAUTE
**Estimation**: Simple

```gitignore
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# Database
*.db
bot_data.db

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/
```

**Critères de validation**:
- [ ] Fichiers sensibles exclus (.env, *.db)
- [ ] Cache Python exclu
- [ ] Environnements virtuels exclus

---

## T0.2 - Configuration de Base

### T0.2.1 - Créer le module de configuration centralisé
**Priorité**: HAUTE
**Estimation**: Simple

Créer `src/config.py` :

```python
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

@dataclass
class Config:
    # LLM
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"

    # Exchange MEXC
    mexc_api_key: str
    mexc_api_secret: str
    # Note: MEXC n'a pas de passphrase

    # Trading
    paper_trading: bool
    base_currency: str

    # Bot
    loop_interval_seconds: int
    max_order_usd: float
    max_equity_pct: float
    max_daily_loss_usd: float
    max_positions: int

def load_config() -> Config:
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        mexc_api_key=os.getenv("MEXC_API_KEY", ""),
        mexc_api_secret=os.getenv("MEXC_API_SECRET", ""),
        paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
        base_currency=os.getenv("BASE_CURRENCY", "USDT"),
        loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "300")),
        max_order_usd=float(os.getenv("MAX_ORDER_USD", "20.0")),
        max_equity_pct=float(os.getenv("MAX_EQUITY_PCT", "0.05")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "50.0")),
        max_positions=int(os.getenv("MAX_POSITIONS", "5")),
    )

# Singleton
config: Optional[Config] = None

def get_config() -> Config:
    global config
    if config is None:
        config = load_config()
    return config
```

**Critères de validation**:
- [ ] Configuration centralisée et typée
- [ ] Valeurs par défaut sécurisées
- [ ] Pattern singleton pour éviter les rechargements

---

### T0.2.2 - Créer le point d'entrée main.py
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `main.py` avec structure de base :

```python
#!/usr/bin/env python3
"""
OtterTrend - Bot de Trading Autonome SocialFi/Crypto
Point d'entrée principal
"""

import asyncio
import sys
from dotenv import load_dotenv

from src.config import get_config
from src.client.groq_adapter import GroqAdapter
from src.bot.loop import TradingBotLoop
from src.ui.renderer import Renderer

# System Prompt pour le LLM (aligné ChatGPT - Bot 100% Autonome)
SYSTEM_PROMPT = """
Tu es OtterTrend, un Bot de Trading 100% AUTONOME.

## MISSION
Maximiser le ROI quotidien (cible >1%) en tradant des narratifs Crypto,
particulièrement SocialFi, sur MEXC.

## CAPACITÉS
Tu as accès à des outils pour :
- OBSERVER : prix, trends Google, mentions sociales, news
- ANALYSER : régime marché, sentiment, force des narratifs
- AGIR : passer des ordres, gérer le portfolio

## RÈGLES ABSOLUES
1. Tu DÉCIDES et AGIS toi-même. Pas de "je recommande".
2. Tu appelles place_order() directement quand tu veux trader.
3. La couche risque ajustera ou rejettera si nécessaire.
4. Tu expliques ton raisonnement AVANT chaque action.
5. Tu ne dépasses JAMAIS les limites de risque codées.

## AVANTAGES MEXC
- Frais 0% maker / 0.01% taker - profite pour scalper
- Listings rapides - tokens dispo avant OKX/Binance
- Surveille les nouveaux listings car c'est la spécialité de cet exchange

## STRATÉGIE (ChatGPT spec)
1. Surveille Google Trends pour détecter les narratifs en hausse
2. Corrèle avec le sentiment social (X, Farcaster)
3. Entre tôt sur les tokens liés au narratif montant
4. Sors agressivement quand le narratif sature
5. Exploite les événements : listings, airdrops, V2

## NARRATIFS À SUIVRE
- SocialFi (Farcaster, Lens, friend.tech, CyberConnect)
- AI Crypto (FET, RNDR, AGIX)
- Memecoins (trends viraux)
- RWA (tokenisation assets réels)

FORMAT DE RÉPONSE :
Retourne un JSON avec la clé "actions" contenant tes décisions:
{ "actions": [
    {"type": "OPEN", "symbol": "TOKEN/USDT", "side": "buy", "size_pct_equity": 0.02},
    {"type": "CLOSE", "symbol": "TOKEN/USDT"},
    {"type": "HOLD", "reason": "..."}
]}
"""


async def main() -> int:
    load_dotenv()
    cfg = get_config()

    # Validation configuration
    if not cfg.groq_api_key:
        print("[ERROR] GROQ_API_KEY manquant dans .env")
        return 1

    if not cfg.paper_trading and not cfg.mexc_api_key:
        print("[ERROR] MEXC_API_KEY requis en mode live")
        return 1

    print(f"[INFO] Démarrage OtterTrend")
    print(f"[INFO] Mode: {'PAPER' if cfg.paper_trading else 'LIVE'}")
    print(f"[INFO] Exchange: MEXC (frais 0%/0.01%)")
    print(f"[INFO] Intervalle: {cfg.loop_interval_seconds}s")

    # Initialisation des composants
    groq_client = GroqAdapter(
        api_key=cfg.groq_api_key,
        model=cfg.llm_model,
        system_prompt=SYSTEM_PROMPT,
    )

    renderer = Renderer()
    bot_loop = TradingBotLoop(
        groq_client=groq_client,
        renderer=renderer,
    )

    # Boucle principale
    try:
        await bot_loop.run_forever(interval_seconds=cfg.loop_interval_seconds)
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé par l'utilisateur")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Critères de validation**:
- [ ] Point d'entrée fonctionnel
- [ ] Validation de configuration au démarrage
- [ ] Messages de status clairs
- [ ] Gestion propre de l'interruption clavier

---

## T0.3 - Base de Données SQLite

### T0.3.1 - Implémenter src/bot/memory.py
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer le module de persistance SQLite3 avec schéma complet :

**Tables requises** :
1. `trades` - Historique des trades
2. `logs` - Logs de décisions et erreurs
3. `config` - Configuration persistante
4. `market_cache` - Cache des données marché (optionnel)

**Méthodes requises** :
- `__init__(db_path)` - Initialise la connexion et le schéma
- `log(level, message, context)` - Log générique
- `log_info/log_error/log_decision` - Helpers de logging
- `log_trade_open(order, snapshot, action)` - Enregistre ouverture de trade
- `log_trade_close(order, snapshot, action)` - Enregistre fermeture de trade
- `get_open_trades()` - Liste des trades ouverts
- `get_trade_history(limit)` - Historique des trades
- `get_pnl_summary()` - Résumé PnL

**Schéma SQL** :
```sql
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    amount REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    timestamp_open DATETIME DEFAULT CURRENT_TIMESTAMP,
    timestamp_close DATETIME,
    pnl REAL,
    pnl_pct REAL,
    status TEXT DEFAULT 'open',
    metadata JSON
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    context_snapshot JSON
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
```

**Critères de validation**:
- [ ] Schéma créé automatiquement à l'init
- [ ] Méthodes CRUD fonctionnelles
- [ ] Gestion propre des connexions (context manager)
- [ ] Index pour performances de requêtes
- [ ] Tests unitaires passants

---

## T0.4 - Adaptateur LLM Groq

### T0.4.1 - Implémenter src/client/groq_adapter.py
**Priorité**: CRITIQUE
**Estimation**: Moyenne

**Fonctionnalités requises** :
1. Interface compatible OpenAI
2. Support du streaming
3. Support du function calling
4. Injection automatique du system prompt
5. Gestion des erreurs et retry

**Interface** :
```python
class GroqAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> None: ...

    def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Generator[Dict, None, None]: ...
```

**Format des événements en streaming** :
```python
{"type": "token", "content": "texte..."}
{"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
{"type": "done"}
{"type": "error", "message": "..."}
```

**Critères de validation**:
- [ ] Streaming fonctionnel
- [ ] Function calling avec parsing JSON
- [ ] System prompt injecté correctement
- [ ] Gestion des erreurs API (rate limit, timeout)
- [ ] Tests avec mocks

---

## T0.5 - Boucle Principale

### T0.5.1 - Implémenter src/bot/loop.py (squelette)
**Priorité**: CRITIQUE
**Estimation**: Moyenne

**Cycle Observe → Think → Act** :

```python
class TradingBotLoop:
    def __init__(
        self,
        groq_client: GroqAdapter,
        renderer: Optional[Renderer] = None,
    ) -> None: ...

    async def run_forever(self, interval_seconds: int = 60) -> None: ...

    async def _observe(self) -> Dict[str, Any]:
        """Collecte: marché + portfolio + trends"""
        ...

    async def _think(self, snapshot: Dict) -> List[Dict]:
        """Envoie au LLM, récupère les actions"""
        ...

    async def _act(self, snapshot: Dict, actions: List[Dict]) -> None:
        """Exécute les actions après validation risk"""
        ...
```

**Critères de validation**:
- [ ] Boucle infinie avec intervalle configurable
- [ ] Gestion des erreurs par cycle (ne crash pas)
- [ ] Logging de chaque phase
- [ ] Intégration avec le renderer (optionnel)

---

## Checklist Finale Phase 0

- [ ] Structure de fichiers complète
- [ ] Configuration centralisée fonctionnelle
- [ ] Base de données SQLite initialisée
- [ ] Adaptateur Groq avec streaming
- [ ] Boucle principale (squelette)
- [ ] Point d'entrée main.py fonctionnel
- [ ] `python main.py` démarre sans erreur (même si pas d'action)

---

## Dépendances

- Aucune dépendance sur d'autres phases
- Pré-requis pour toutes les phases suivantes

## Notes Techniques

- Python 3.10+ requis pour les features async modernes
- Utiliser `asyncio` pour toutes les opérations I/O
- Préférer les dataclasses aux dicts pour le typage
- Documenter toutes les fonctions publiques avec docstrings

## Notes MEXC

- **Pas de passphrase** (contrairement à OKX) - juste API key + secret
- **recvWindow**: Requis dans les options CCXT, typiquement 60000ms
- **Rate limits plus stricts** - ajouter enableRateLimit=True
- **Frais 0% maker** - optimiser pour ordres limite quand possible

---

## T0.6 - Architecture Modulaire (Interfaces & Abstractions)

> **Objectif**: Assurer que chaque composant est interchangeable via des interfaces abstraites.
> Cela permet de swapper facilement l'exchange (MEXC → Binance), le LLM (Groq → OpenAI), etc.

### T0.6.1 - Créer les interfaces de base
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `src/interfaces.py` avec les protocoles/ABC pour tous les composants :

```python
"""
Interfaces abstraites pour la modularité du bot.
Chaque composant majeur doit implémenter une interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass


# === EXCHANGE INTERFACE ===

class BaseExchange(ABC):
    """
    Interface abstraite pour un exchange.
    Permet de swapper MEXC ↔ Binance ↔ OKX ↔ Paper Trading.
    """

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Récupère le ticker pour un symbole"""
        ...

    @abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """Récupère les tickers pour plusieurs symboles"""
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Récupère le carnet d'ordres"""
        ...

    @abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """Récupère la balance du compte"""
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place un ordre"""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Annule un ordre"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Ferme la connexion"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de l'exchange (mexc, binance, paper, etc.)"""
        ...

    @property
    @abstractmethod
    def fees(self) -> Dict[str, float]:
        """Retourne les frais {maker: x, taker: y}"""
        ...


# === LLM INTERFACE ===

class BaseLLMAdapter(ABC):
    """
    Interface abstraite pour un provider LLM.
    Permet de swapper Groq ↔ OpenAI ↔ Anthropic ↔ Local.
    """

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Generator[Dict, None, None]:
        """Stream une réponse chat avec support function calling"""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les infos du modèle (nom, limites, etc.)"""
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Indique si le modèle supporte les function calls"""
        ...


# === DATA PROVIDER INTERFACES ===

class BaseTrendsProvider(ABC):
    """
    Interface pour les providers de tendances.
    Permet de swapper Google Trends ↔ Autre source.
    """

    @abstractmethod
    async def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "now 7-d",
    ) -> Dict[str, Any]:
        """Récupère l'intérêt dans le temps pour des mots-clés"""
        ...

    @abstractmethod
    async def get_related_queries(self, keyword: str) -> Dict[str, List[str]]:
        """Récupère les recherches associées"""
        ...


class BaseNewsProvider(ABC):
    """
    Interface pour les providers de news.
    Permet de swapper CryptoCompare ↔ CoinGecko ↔ RSS.
    """

    @abstractmethod
    async def get_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les dernières news"""
        ...

    @abstractmethod
    async def search_news_by_symbol(
        self, symbol: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Recherche les news pour un symbole"""
        ...


class BaseSentimentAnalyzer(ABC):
    """
    Interface pour l'analyse de sentiment.
    Permet de swapper règles simples ↔ FinBERT ↔ GPT.
    """

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyse le sentiment d'un texte"""
        ...

    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyse le sentiment de plusieurs textes"""
        ...


# === RISK INTERFACE ===

@dataclass
class OrderRequest:
    """Requête d'ordre à valider"""
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None
    order_type: str = "market"


@dataclass
class RiskCheckResult:
    """Résultat d'une validation de risque"""
    approved: bool
    adjusted_amount: Optional[float] = None
    reason: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class BaseRiskManager(ABC):
    """
    Interface pour le gestionnaire de risque.
    Permet de définir des règles custom ou de swapper l'implémentation.
    """

    @abstractmethod
    def check_order(
        self,
        order: OrderRequest,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> RiskCheckResult:
        """Vérifie si un ordre respecte les règles de risque"""
        ...

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """Retourne les contraintes de risque actuelles"""
        ...

    @abstractmethod
    def update_daily_stats(self, pnl: float) -> None:
        """Met à jour les stats journalières (PnL, trades, etc.)"""
        ...

    @abstractmethod
    def should_halt(self) -> bool:
        """Indique si le trading doit être arrêté (daily loss, etc.)"""
        ...


# === PERSISTENCE INTERFACE ===

class BaseMemory(ABC):
    """
    Interface pour la persistance des données.
    Permet de swapper SQLite ↔ PostgreSQL ↔ Redis.
    """

    @abstractmethod
    def log(self, level: str, message: str, context: Optional[Dict] = None) -> None:
        """Log un message"""
        ...

    @abstractmethod
    def log_trade_open(self, order: Dict, snapshot: Dict, action: Dict) -> int:
        """Enregistre l'ouverture d'un trade, retourne l'ID"""
        ...

    @abstractmethod
    def log_trade_close(self, trade_id: int, order: Dict, pnl: float) -> None:
        """Enregistre la fermeture d'un trade"""
        ...

    @abstractmethod
    def get_open_trades(self) -> List[Dict]:
        """Retourne les trades ouverts"""
        ...

    @abstractmethod
    def get_daily_pnl(self) -> float:
        """Retourne le PnL du jour"""
        ...


# === TOOL INTERFACE ===

@dataclass
class ToolDefinition:
    """Définition d'un tool pour le LLM"""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str  # "observer", "reflechir", "agir"


class BaseTool(ABC):
    """
    Interface pour un tool appelable par le LLM.
    Permet d'ajouter facilement de nouveaux tools.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Retourne la définition du tool pour le LLM"""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Exécute le tool avec les arguments fournis"""
        ...
```

**Critères de validation**:
- [ ] Toutes les interfaces ABC définies
- [ ] Dataclasses pour les structures de données
- [ ] Documentation claire de chaque méthode
- [ ] Typage strict

---

### T0.6.2 - Créer le registre de tools (Plugin System)
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `src/tools/registry.py` pour enregistrer dynamiquement les tools :

```python
"""
Registre de tools - système de plugins.
Permet d'ajouter/retirer des tools dynamiquement.
"""

from typing import Dict, List, Optional, Type
from src.interfaces import BaseTool, ToolDefinition


class ToolRegistry:
    """
    Registre central pour tous les tools du bot.
    Pattern Singleton avec enregistrement dynamique.
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """Enregistre un tool dans le registre"""
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' déjà enregistré")
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """Retire un tool du registre"""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[BaseTool]:
        """Récupère un tool par son nom"""
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """Retourne tous les tools enregistrés"""
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[BaseTool]:
        """Retourne les tools d'une catégorie (observer, reflechir, agir)"""
        return [
            tool for tool in self._tools.values()
            if tool.definition.category == category
        ]

    def get_schemas(self) -> List[Dict]:
        """Retourne les schemas JSON de tous les tools pour le LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.definition.name,
                    "description": tool.definition.description,
                    "parameters": tool.definition.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> Dict:
        """Exécute un tool par son nom"""
        tool = self.get(name)
        if tool is None:
            return {"error": f"Tool '{name}' non trouvé"}
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return {"error": str(e)}

    def clear(self) -> None:
        """Vide le registre (utile pour les tests)"""
        self._tools.clear()


# Singleton global
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Retourne le registre singleton"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


# Décorateur pour enregistrer un tool facilement
def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    Décorateur pour auto-enregistrer un tool.

    Usage:
        @register_tool
        class MyTool(BaseTool):
            ...
    """
    instance = cls()
    get_tool_registry().register(instance)
    return cls
```

**Critères de validation**:
- [ ] Pattern singleton fonctionnel
- [ ] Méthodes CRUD pour les tools
- [ ] Décorateur pour auto-enregistrement
- [ ] Génération des schemas JSON
- [ ] Tests unitaires

---

### T0.6.3 - Créer le conteneur d'injection de dépendances
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `src/container.py` pour l'injection de dépendances :

```python
"""
Conteneur d'injection de dépendances.
Permet de configurer et swapper les implémentations facilement.
"""

from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import dataclass, field

from src.interfaces import (
    BaseExchange,
    BaseLLMAdapter,
    BaseTrendsProvider,
    BaseNewsProvider,
    BaseSentimentAnalyzer,
    BaseRiskManager,
    BaseMemory,
)

T = TypeVar("T")


@dataclass
class Container:
    """
    Conteneur IoC (Inversion of Control).
    Centralise toutes les dépendances du bot.
    """

    # Instances des composants
    exchange: Optional[BaseExchange] = None
    llm: Optional[BaseLLMAdapter] = None
    trends_provider: Optional[BaseTrendsProvider] = None
    news_provider: Optional[BaseNewsProvider] = None
    sentiment_analyzer: Optional[BaseSentimentAnalyzer] = None
    risk_manager: Optional[BaseRiskManager] = None
    memory: Optional[BaseMemory] = None

    # Factories pour lazy initialization
    _factories: Dict[str, callable] = field(default_factory=dict)

    def register_factory(self, name: str, factory: callable) -> None:
        """Enregistre une factory pour création lazy"""
        self._factories[name] = factory

    def get_or_create(self, name: str) -> Any:
        """Récupère ou crée une instance via factory"""
        current = getattr(self, name, None)
        if current is not None:
            return current

        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"Pas de factory pour '{name}'")

        instance = factory()
        setattr(self, name, instance)
        return instance

    def validate(self) -> bool:
        """Vérifie que toutes les dépendances requises sont configurées"""
        required = ["exchange", "llm", "risk_manager", "memory"]
        missing = [
            name for name in required
            if getattr(self, name, None) is None and name not in self._factories
        ]
        if missing:
            raise ValueError(f"Dépendances manquantes: {missing}")
        return True


# Singleton global
_container: Optional[Container] = None


def get_container() -> Container:
    """Retourne le conteneur singleton"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def configure_container(
    exchange: Optional[BaseExchange] = None,
    llm: Optional[BaseLLMAdapter] = None,
    trends_provider: Optional[BaseTrendsProvider] = None,
    news_provider: Optional[BaseNewsProvider] = None,
    sentiment_analyzer: Optional[BaseSentimentAnalyzer] = None,
    risk_manager: Optional[BaseRiskManager] = None,
    memory: Optional[BaseMemory] = None,
) -> Container:
    """
    Configure le conteneur avec les implémentations.

    Usage:
        from src.container import configure_container
        from src.tools.market import MEXCExchange
        from src.client.groq_adapter import GroqAdapter

        configure_container(
            exchange=MEXCExchange(),
            llm=GroqAdapter(api_key=..., model=...),
            ...
        )
    """
    container = get_container()
    if exchange:
        container.exchange = exchange
    if llm:
        container.llm = llm
    if trends_provider:
        container.trends_provider = trends_provider
    if news_provider:
        container.news_provider = news_provider
    if sentiment_analyzer:
        container.sentiment_analyzer = sentiment_analyzer
    if risk_manager:
        container.risk_manager = risk_manager
    if memory:
        container.memory = memory
    return container


def reset_container() -> None:
    """Reset le conteneur (utile pour les tests)"""
    global _container
    _container = None
```

**Critères de validation**:
- [ ] Pattern IoC fonctionnel
- [ ] Support des factories lazy
- [ ] Validation des dépendances requises
- [ ] Fonction de configuration facile
- [ ] Reset pour les tests

---

### T0.6.4 - Adapter main.py pour l'injection de dépendances
**Priorité**: HAUTE
**Estimation**: Simple

Modifier `main.py` pour utiliser le conteneur :

```python
#!/usr/bin/env python3
"""
OtterTrend - Bot de Trading Autonome SocialFi/Crypto
Point d'entrée principal avec injection de dépendances.
"""

import asyncio
import sys
from dotenv import load_dotenv

from src.config import get_config
from src.container import configure_container, get_container

# Implémentations concrètes (facilement swappables)
from src.client.groq_adapter import GroqAdapter
from src.tools.market import MEXCExchange, PaperExchange
from src.tools.trends import GoogleTrendsProvider
from src.tools.news import CryptoCompareProvider
from src.tools.sentiment import RuleBasedSentiment
from src.tools.risk import DefaultRiskManager
from src.bot.memory import SQLiteMemory
from src.bot.loop import TradingBotLoop
from src.ui.renderer import Renderer

SYSTEM_PROMPT = """..."""  # Inchangé


def setup_container() -> None:
    """
    Configure le conteneur avec les implémentations.
    Modifiez cette fonction pour swapper les composants.
    """
    cfg = get_config()

    # Exchange: MEXC ou Paper selon config
    if cfg.paper_trading:
        exchange = PaperExchange(initial_balance=1000.0)
    else:
        exchange = MEXCExchange()

    # LLM: Groq (peut être swappé pour OpenAI, etc.)
    llm = GroqAdapter(
        api_key=cfg.groq_api_key,
        model=cfg.llm_model,
        system_prompt=SYSTEM_PROMPT,
    )

    # Data providers
    trends = GoogleTrendsProvider()
    news = CryptoCompareProvider()
    sentiment = RuleBasedSentiment()

    # Risk & Memory
    risk = DefaultRiskManager(cfg)
    memory = SQLiteMemory(db_path="bot_data.db")

    # Configuration du conteneur
    configure_container(
        exchange=exchange,
        llm=llm,
        trends_provider=trends,
        news_provider=news,
        sentiment_analyzer=sentiment,
        risk_manager=risk,
        memory=memory,
    )


async def main() -> int:
    load_dotenv()
    cfg = get_config()

    # Validation de base
    if not cfg.groq_api_key:
        print("[ERROR] GROQ_API_KEY manquant dans .env")
        return 1

    # Setup du conteneur IoC
    setup_container()
    container = get_container()
    container.validate()

    print(f"[INFO] Démarrage OtterTrend")
    print(f"[INFO] Mode: {'PAPER' if cfg.paper_trading else 'LIVE'}")
    print(f"[INFO] Exchange: {container.exchange.name}")
    print(f"[INFO] LLM: {container.llm.get_model_info()['name']}")

    # Boucle principale avec dépendances injectées
    renderer = Renderer()
    bot_loop = TradingBotLoop(container=container, renderer=renderer)

    try:
        await bot_loop.run_forever(interval_seconds=cfg.loop_interval_seconds)
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Critères de validation**:
- [ ] Setup du conteneur centralisé
- [ ] Composants facilement swappables
- [ ] Validation au démarrage
- [ ] Logs informatifs sur les composants utilisés
