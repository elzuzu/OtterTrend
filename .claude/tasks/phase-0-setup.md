# Phase 0 - Setup & Architecture de Base

> **Objectif**: Mettre en place le repository, la structure de fichiers et les configurations de base pour le bot OtterTrend.

## Statut Global
- [ ] Phase complète

---

## T0.1 - Structure du Projet

### T0.1.1 - Créer l'arborescence de fichiers
**Priorité**: CRITIQUE
**Estimation**: Simple

Créer la structure suivante :
```
OtterTrend/
├── main.py                          # Point d'entrée (boucle infinie)
├── requirements.txt                 # Dépendances Python
├── .env.example                     # Template des variables d'environnement
├── .gitignore                       # Fichiers à ignorer
├── src/
│   ├── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   └── groq_adapter.py         # Adaptateur LLM Groq
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── brain.py                # Logique décisionnelle LLM
│   │   ├── memory.py               # Persistance SQLite3
│   │   └── loop.py                 # Orchestrateur Observe→Think→Act
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── market.py               # Interface MEXC/CCXT
│   │   ├── trends.py               # Google Trends + sentiment
│   │   ├── risk.py                 # Garde-fous hard-coded
│   │   ├── analytics.py            # ML/Stats basiques
│   │   └── schemas.py              # Définitions tools LLM
│   └── ui/
│       ├── __init__.py
│       └── renderer.py             # Visualisation Rich CLI
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
# === LLM Provider ===
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# === Exchange Configuration ===
EXCHANGE_ID=mexc                    # mexc | bybit | okx
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
EXCHANGE_TESTNET=true               # true pour testnet, false pour production

# === Trading Mode ===
PAPER_TRADING=true                  # true = simulation, false = réel
BASE_CURRENCY=USDT

# === Bot Settings ===
LOOP_INTERVAL_SECONDS=60            # Intervalle entre les cycles
MAX_ORDER_USD=20.0                  # Limite absolue par ordre
MAX_EQUITY_PCT=0.05                 # Max 5% du portefeuille par trade
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

    # Exchange
    exchange_id: str
    exchange_api_key: str
    exchange_api_secret: str
    exchange_testnet: bool

    # Trading
    paper_trading: bool
    base_currency: str

    # Bot
    loop_interval_seconds: int
    max_order_usd: float
    max_equity_pct: float

def load_config() -> Config:
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        exchange_id=os.getenv("EXCHANGE_ID", "mexc"),
        exchange_api_key=os.getenv("EXCHANGE_API_KEY", ""),
        exchange_api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
        exchange_testnet=os.getenv("EXCHANGE_TESTNET", "true").lower() == "true",
        paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
        base_currency=os.getenv("BASE_CURRENCY", "USDT"),
        loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
        max_order_usd=float(os.getenv("MAX_ORDER_USD", "20.0")),
        max_equity_pct=float(os.getenv("MAX_EQUITY_PCT", "0.05")),
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

# System Prompt pour le LLM
SYSTEM_PROMPT = """
Tu es OtterTrend, un Bot de Trading Autonome spécialisé dans les narratifs SocialFi et Crypto Trends.
Tu trades sur MEXC (via CCXT) avec un objectif de ROI journalier > 1%.

RÈGLES DE CONDUITE :
1. Observe les données marché, tendances Google, et sentiment social via tes outils.
2. Décide quand ouvrir, ajuster ou fermer des positions.
3. Respecte TOUJOURS les garde-fous de risque (max $20/ordre, max 5% equity).
4. Explique ton raisonnement de façon concise avant chaque action.
5. Exploite les frais bas MEXC pour capturer des micro-mouvements.
6. Surveille les nouveaux listings et top gainers pour des opportunités "early trend".
7. Évite les ordres trop gros sur les pairs peu liquides.

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

    if not cfg.paper_trading and not cfg.exchange_api_key:
        print("[ERROR] EXCHANGE_API_KEY requis en mode live")
        return 1

    print(f"[INFO] Démarrage OtterTrend")
    print(f"[INFO] Mode: {'PAPER' if cfg.paper_trading else 'LIVE'}")
    print(f"[INFO] Exchange: {cfg.exchange_id}")
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
