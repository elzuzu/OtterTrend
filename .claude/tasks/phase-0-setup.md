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
