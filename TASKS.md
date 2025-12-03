# OtterTrend - Plan de Développement

> **Objectif**: Bot de trading 100% AUTONOME SocialFi/Crypto
>
> **Exchange Principal**: MEXC (frais bas, listings rapides)
>
> **Technologie LLM**: Groq (Llama 3.3 70B Versatile)
>
> **ROI Cible**: >1% journalier
>
> **Architecture**: LLM Orchestrateur + Tools Observer/Réfléchir/Agir

---

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

---

## Vue d'Ensemble des Phases

| Phase | Nom | Description | Priorité | Statut |
|-------|-----|-------------|----------|--------|
| 0 | [Setup & Architecture](/.claude/tasks/phase-0-setup.md) | Structure, config, base de données | CRITIQUE | 🟢 |
| 1 | [Market & Portfolio](/.claude/tasks/phase-1-market.md) | Interface MEXC/CCXT | CRITIQUE | 🔴 |
| 2 | [Trends & Social](/.claude/tasks/phase-2-trends.md) | Google Trends, sentiment news | HAUTE | 🔴 |
| 3 | [UI Rich](/.claude/tasks/phase-3-ui.md) | Interface terminal Rich | MOYENNE | 🔴 |
| 4 | [Function Calling](/.claude/tasks/phase-4-function-calling.md) | Tools LLM, multi-tour | HAUTE | 🔴 |
| 5 | [Risk Management](/.claude/tasks/phase-5-risk.md) | Garde-fous, paper trading | CRITIQUE | 🔴 |
| 6 | [Tests & Hardening](/.claude/tasks/phase-6-testing.md) | Tests, sécurité, déploiement | HAUTE | 🔴 |

**Légende**: 🔴 Non commencé | 🟡 En cours | 🟢 Complété

---

## Dépendances entre Phases

```
Phase 0 (Setup)
    │
    ├── Phase 1 (Market) ──┐
    │                      │
    ├── Phase 2 (Trends) ──┼── Phase 4 (Function Calling)
    │                      │          │
    └── Phase 3 (UI) ──────┘          │
                                      │
                           Phase 5 (Risk) ←─┘
                                      │
                           Phase 6 (Tests & Deploy)
```

---

## Résumé des Tâches par Phase

### Phase 0 - Setup & Architecture de Base
**Fichier**: [.claude/tasks/phase-0-setup.md](/.claude/tasks/phase-0-setup.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T0.1.1 | Créer l'arborescence de fichiers | CRITIQUE | 🟢 |
| T0.1.2 | Créer requirements.txt | CRITIQUE | 🟢 |
| T0.1.3 | Créer .env.example | HAUTE | 🟢 |
| T0.1.4 | Créer .gitignore | HAUTE | 🟢 |
| T0.2.1 | Module de configuration centralisé | HAUTE | 🟢 |
| T0.2.2 | Point d'entrée main.py | CRITIQUE | 🟢 |
| T0.3.1 | Implémenter src/bot/memory.py (SQLite) | CRITIQUE | 🟢 |
| T0.4.1 | Implémenter src/client/groq_adapter.py | CRITIQUE | 🟢 |
| T0.5.1 | Implémenter src/bot/loop.py (squelette) | CRITIQUE | 🟢 |
| T0.7.1 | Setup script Mac Mini M4 ARM64 | HAUTE | 🟢 |
| T0.7.2 | Module détection hardware (src/hardware.py) | HAUTE | 🟢 |
| T0.7.3 | Requirements Apple Silicon (MLX, Core ML) | HAUTE | 🟢 |
| T0.7.4 | Interfaces accélérateurs hardware | HAUTE | 🟢 |
| T0.7.5 | Backend MLX (src/accelerators/mlx_backend.py) | HAUTE | 🟢 |
| T0.7.6 | Backend Core ML (src/accelerators/coreml_backend.py) | HAUTE | 🟢 |

---

### Phase 1 - Market & Portfolio (MEXC via CCXT)
**Fichier**: [.claude/tasks/phase-1-market.md](/.claude/tasks/phase-1-market.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T1.1.1 | Wrapper CCXT multi-exchange | CRITIQUE | 🟢 |
| T1.1.2 | Méthodes de lecture marché | CRITIQUE | 🟢 |
| T1.2.1 | Méthodes de lecture portefeuille | CRITIQUE | 🟢 |
| T1.2.2 | Snapshot marché complet | HAUTE | 🟢 |
| T1.3.1 | Passation d'ordres | CRITIQUE | 🔴 |
| T1.4.1 | Détection nouveaux listings | HAUTE | 🔴 |
| T1.4.2 | Scan top gainers | HAUTE | 🔴 |
| T1.4.3 | Estimation de liquidité | MOYENNE | 🔴 |
| T1.5.1 | Simulateur Paper Trading | CRITIQUE | 🔴 |
| T1.5.2 | Factory exchange client | HAUTE | 🔴 |
| T1.7.1 | Stack réseau "Envoy-like" (HTTP/2, pool, keep-alive) | CRITIQUE | 🔴 |
| T1.7.2 | Circuit breaker, retries, hedging, outlier detection | CRITIQUE | 🔴 |
| T1.7.3 | Limiteurs & QoS adaptatifs (token bucket, priorité) | HAUTE | 🔴 |
| T1.7.4 | Health-checks actifs + failover multi-endpoints | HAUTE | 🔴 |
| T1.7.5 | Observabilité OpenTelemetry (latence, saturation) | HAUTE | 🔴 |

---

### Phase 2 - Trends & Social Sentiment
**Fichier**: [.claude/tasks/phase-2-trends.md](/.claude/tasks/phase-2-trends.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T2.1.1 | Wrapper PyTrends | CRITIQUE | 🔴 |
| T2.1.2 | Keywords crypto/SocialFi | HAUTE | 🔴 |
| T2.2.1 | Fetcher news crypto | HAUTE | 🔴 |
| T2.2.2 | Analyse de sentiment basique | HAUTE | 🔴 |
| T2.3.1 | Architecture Twitter (stub) | BASSE | 🔴 |
| T2.4.1 | Snapshot tendances unifié | CRITIQUE | 🔴 |
| T2.5.1 | Analytics basiques | MOYENNE | 🔴 |
| T2.7.1 | MLXSentimentAnalyzer (Apple Silicon) | HAUTE | 🔴 |
| T2.7.2 | CoreMLSentimentAnalyzer (Neural Engine) | HAUTE | 🔴 |
| T2.7.3 | Accélération calculs vectoriels MLX | MOYENNE | 🔴 |
| T2.7.4 | TrendAnalyzer avec auto-backend hardware | MOYENNE | 🔴 |
| T2.7.5 | Script benchmark backends sentiment | BASSE | 🔴 |
| T2.7.6 | Factory backends hardware | HAUTE | 🔴 |

---

### Phase 3 - UI Rich / Renderer CLI
**Fichier**: [.claude/tasks/phase-3-ui.md](/.claude/tasks/phase-3-ui.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T3.1.1 | Layout principal | HAUTE | 🔴 |
| T3.1.2 | Structure base Renderer | CRITIQUE | 🔴 |
| T3.2.1 | Méthodes de mise à jour | CRITIQUE | 🔴 |
| T3.3.1 | Mode Live avec Rich | HAUTE | 🔴 |
| T3.3.2 | Mode simplifié (debug) | MOYENNE | 🔴 |
| T3.4.1 | Intégration TradingBotLoop | HAUTE | 🔴 |
| T3.4.2 | Activation dans main.py | HAUTE | 🔴 |
| T3.5.1 | Spinners et loading | BASSE | 🔴 |
| T3.5.2 | Barres de progression | BASSE | 🔴 |

---

### Phase 4 - LLM Function Calling
**Fichier**: [.claude/tasks/phase-4-function-calling.md](/.claude/tasks/phase-4-function-calling.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T4.1.1 | Schemas JSON des tools | CRITIQUE | 🔴 |
| T4.2.1 | Router de tools | CRITIQUE | 🔴 |
| T4.3.1 | Multi-tour GroqAdapter | HAUTE | 🔴 |
| T4.3.2 | Intégration loop.py | HAUTE | 🔴 |
| T4.4.1 | Cache des résultats | MOYENNE | 🔴 |
| T4.4.2 | Rate limiting tools | MOYENNE | 🔴 |
| T4.5.1 | Tests unitaires tools | HAUTE | 🔴 |

---

### Phase 5 - Paper Trading & Risk Management
**Fichier**: [.claude/tasks/phase-5-risk.md](/.claude/tasks/phase-5-risk.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T5.1.1 | Risk Manager complet | CRITIQUE | 🔴 |
| T5.2.1 | Système d'alertes | HAUTE | 🔴 |
| T5.3.1 | Améliorer PaperExchangeClient | HAUTE | 🔴 |
| T5.4.1 | Validation pre-live | HAUTE | 🔴 |
| T5.5.1 | Intégration complète | CRITIQUE | 🔴 |

---

### Phase 6 - Tests, Hardening & Deployment
**Fichier**: [.claude/tasks/phase-6-testing.md](/.claude/tasks/phase-6-testing.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T6.1.1 | Tests config | HAUTE | 🔴 |
| T6.1.2 | Tests memory | HAUTE | 🔴 |
| T6.1.3 | Tests risk manager | CRITIQUE | 🔴 |
| T6.1.4 | Tests market | HAUTE | 🔴 |
| T6.2.1 | Tests d'intégration | HAUTE | 🔴 |
| T6.3.1 | Audit de sécurité | CRITIQUE | 🔴 |
| T6.3.2 | Hardening config | HAUTE | 🔴 |
| T6.4.1 | README.md | HAUTE | 🔴 |
| T6.4.2 | Docstrings modules | MOYENNE | 🔴 |
| T6.5.1 | Dockerfile | MOYENNE | 🔴 |
| T6.5.2 | docker-compose.yml | MOYENNE | 🔴 |
| T6.5.3 | Script de démarrage | HAUTE | 🔴 |


---

## Rappels MVP (à respecter partout)

* Exchange par défaut : **MEXC spot USDT**, paramétrable via `EXCHANGE_ID` (`mexc` par défaut). Prévoir `EXCHANGE_TESTNET` et `PAPER_TRADING`.
* LLM MVP **sans function calling** : entrée `SNAPSHOT_JSON`, sortie JSON `{ "actions": [...] }` (types `OPEN`/`CLOSE`, champ `size_pct_equity` requis pour `OPEN`).
* Trends : Google Trends mots-clés `socialfi`, `crypto airdrop`, `memecoin` + nouveaux listings MEXC + top gainers 24h MEXC. Sentiment social avancé = V2.
* SQLite : tables `trades`, `logs`, `config` uniquement pour le MVP.
* UI Rich : 3 panneaux minimum (Thoughts / Actions / Portfolio basique). Tout le reste en V2.
* Risque :
  * Limites hard : `RISK_MAX_TRADE_USD` (20 par défaut) et `RISK_MAX_TRADE_PCT_EQUITY` (0.05 par défaut).
  * Filtre liquidité : helper `estimate_pair_liquidity(symbol)` qui bloque les paires < `RISK_MIN_LIQUIDITY_USD` et plafonne les ordres à `RISK_LOW_LIQUIDITY_CAP_USD` en cas de volume faible.

---

## Architecture Cible

### Pattern Observer → Réfléchir → Agir (ChatGPT Spec)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ORCHESTRATEUR (Groq)                     │
│                  Llama 3.3 70B Versatile                        │
│            "Tu es OtterTrend, bot 100% AUTONOME"                │
└─────────────────┬───────────────────────────────┬───────────────┘
                  │                               │
    ┌─────────────▼─────────────┐   ┌────────────▼────────────┐
    │       OBSERVER            │   │       RÉFLÉCHIR         │
    │   (Données brutes)        │   │     (mini-ML)           │
    ├───────────────────────────┤   ├─────────────────────────┤
    │ • get_market_snapshot     │   │ • ml_detect_regime      │
    │ • get_orderbook           │   │ • ml_forecast_volatility│
    │ • get_google_trends       │   │ • ml_score_sentiment    │
    │ • get_trending_tokens     │   │ • ml_narrative_strength │
    │ • get_social_mentions     │   │ • ml_estimate_slippage  │
    │ • get_crypto_news         │   │ • ml_detect_anomalies   │
    └───────────────────────────┘   └─────────────────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │          AGIR             │
                    │    (Portfolio & Risk)     │
                    ├───────────────────────────┤
                    │ • get_portfolio_state     │
                    │ • risk_constraints        │
                    │ • risk_check_order        │
                    │ • place_order (MEXC)      │
                    │ • close_position          │
                    │ • cancel_order            │
                    └───────────────────────────┘
```

### Structure de Fichiers

```
OtterTrend/
├── main.py                          # Point d'entrée (boucle autonome)
├── requirements.txt
├── .env.example
├── .gitignore
├── TASKS.md                         # Ce fichier
├── README.md
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration centralisée
│   ├── security.py                  # Validations de sécurité
│   ├── client/
│   │   ├── __init__.py
│   │   └── groq_adapter.py          # Adaptateur LLM Groq
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── brain.py                 # Policy LLM autonome
│   │   ├── memory.py                # SQLite persistence
│   │   └── loop.py                  # Orchestrateur Observe→Think→Act
│   ├── tools/
│   │   ├── __init__.py
│   │   │
│   │   ├── # === OBSERVER (données brutes & trends) ===
│   │   ├── market.py                # get_market_snapshot, get_orderbook (MEXC)
│   │   ├── trends.py                # get_google_trends, get_trending_tokens
│   │   ├── social.py                # get_social_mentions, get_social_trending
│   │   ├── news.py                  # get_crypto_news, get_project_announcements
│   │   │
│   │   ├── # === RÉFLÉCHIR (mini-ML spécialisés) ===
│   │   ├── analytics.py             # ml_detect_regime, ml_forecast_volatility
│   │   ├── sentiment.py             # ml_score_sentiment, ml_narrative_strength
│   │   │
│   │   ├── # === AGIR (portfolio, risk, exécution) ===
│   │   ├── portfolio.py             # get_portfolio_state, risk_constraints
│   │   ├── risk.py                  # risk_check_order (garde-fous hard-coded)
│   │   ├── execution.py             # place_order, close_position (MEXC)
│   │   │
│   │   ├── schemas.py               # Tools JSON schemas pour Groq
│   │   └── router.py                # Tool execution router
│   └── ui/
│       ├── __init__.py
│       └── renderer.py              # Rich CLI (style gemini-cli)
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_memory.py
│   ├── test_risk.py
│   ├── test_market.py
│   └── test_integration.py
├── scripts/
│   └── start.py                     # Script de démarrage avec checks
├── bot_data.db                      # SQLite (runtime)
└── .claude/
    └── tasks/
        ├── phase-0-setup.md
        ├── phase-1-market.md
        ├── phase-2-trends.md
        ├── phase-3-ui.md
        ├── phase-4-function-calling.md
        ├── phase-5-risk.md
        └── phase-6-testing.md
```

---

## Stack Technique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Language | Python | 3.10+ |
| LLM | Groq (Llama 3.3 70B) | Latest |
| Exchange | MEXC via CCXT | 4.0+ |
| Trends | PyTrends | 4.9+ |
| UI | Rich | 13.0+ |
| Database | SQLite3 | Built-in |
| Tests | Pytest | 7.4+ |

---

## Hardware Cible: Mac Mini M4 2024

> **Objectif**: Exploiter nativement les capacités hardware du Mac Mini M4 pour des performances optimales.

### Spécifications M4

| Composant | Spec M4 | Utilisation OtterTrend |
|-----------|---------|------------------------|
| **CPU** | 10-core (4P + 6E) @ 4.4GHz | Async I/O, orchestration |
| **GPU** | 10-core Metal | MLX inference, calculs vectoriels |
| **Neural Engine** | 16-core, 38 TOPS | Core ML sentiment analysis |
| **RAM** | 16-64GB Unified | Zero-copy ML inference |
| **Bandwidth** | 120 GB/s (273 GB/s M4 Pro) | Large batch processing |

### Frameworks Apple Silicon

| Framework | Usage | Avantage |
|-----------|-------|----------|
| **MLX** | Sentiment analysis, embeddings | Zero-copy unified memory, lazy eval |
| **Core ML** | FinBERT inference | Neural Engine 38 TOPS, basse latence |
| **Metal/MPS** | PyTorch fallback | GPU acceleration |
| **Accelerate/vecLib** | NumPy operations | BLAS/LAPACK optimisé Apple |

### Performance Attendue

| Opération | CPU Baseline | Avec Hardware M4 | Speedup |
|-----------|--------------|------------------|---------|
| Sentiment (100 news) | ~2000ms | ~200ms (MLX) | **10x** |
| Cosine similarity (10K vectors) | ~50ms | ~5ms (MLX) | **10x** |
| FinBERT inference | ~500ms | ~50ms (Core ML) | **10x** |
| RSI/Volatility batch | ~10ms | ~2ms (MLX) | **5x** |

### Fichiers Hardware

```
src/
├── hardware.py              # Détection M4, capabilities
├── accelerators/
│   ├── __init__.py
│   ├── mlx_backend.py       # MLX array operations
│   ├── mlx_sentiment.py     # MLXSentimentAnalyzer
│   └── coreml_sentiment.py  # CoreMLSentimentAnalyzer
├── models/
│   └── finbert_sentiment.mlpackage  # Core ML model
scripts/
├── setup_m4.sh              # Setup Python ARM64 optimisé
├── convert_to_coreml.py     # Conversion HuggingFace → Core ML
└── benchmark_sentiment.py   # Benchmark backends

---

## Architecture Modulaire

> **Principe**: Chaque composant majeur est interchangeable via des interfaces abstraites.
> Cela permet de swapper facilement l'exchange, le LLM, les providers de données, etc.

### Interfaces Abstraites (Phase 0)

| Interface | Implémentation par défaut | Alternatives possibles |
|-----------|---------------------------|------------------------|
| `BaseExchange` | MEXCExchange | BinanceExchange, OKXExchange, PaperExchange |
| `BaseLLMAdapter` | GroqAdapter | OpenAIAdapter, AnthropicAdapter, LocalLLM |
| `BaseTrendsProvider` | GoogleTrendsProvider | TwitterTrendsProvider |
| `BaseNewsProvider` | CryptoCompareProvider | CoinGeckoProvider, RSSProvider |
| `BaseSentimentAnalyzer` | RuleBasedSentiment | **MLXSentiment**, **CoreMLSentiment**, FinBERTSentiment |
| `BaseRiskManager` | DefaultRiskManager | ConservativeRiskManager, AggressiveRiskManager |
| `BaseMemory` | SQLiteMemory | PostgresMemory, RedisMemory |
| `BaseTool` | (tous les tools) | Custom tools |
| `BaseMLAccelerator` | MLXAccelerator (M4) | NumPyAccelerator (fallback) |
| `BaseNeuralEngineModel` | CoreMLModel (M4) | PyTorchModel (fallback) |
| `BaseVectorStore` | MLXVectorStore (M4) | NumPyVectorStore (fallback) |

### Patterns de Modularité

1. **Interfaces ABC** (`src/interfaces.py`)
   - Définit les contrats pour chaque composant
   - Permet le type-checking et l'IDE support

2. **Container IoC** (`src/container.py`)
   - Injection de dépendances centralisée
   - Factories pour création lazy des instances

3. **Registres de Plugins**
   - `EXCHANGE_REGISTRY` - Exchanges disponibles
   - `ToolRegistry` - Tools enregistrés dynamiquement
   - `TRENDS_REGISTRY`, `NEWS_REGISTRY`, `SENTIMENT_REGISTRY`

4. **Pattern Chain of Responsibility** (Risk Rules)
   - Règles de risque indépendantes et testables
   - Facile d'ajouter/retirer des règles

### Comment swapper un composant

```python
from src.container import configure_container
from src.tools.market import BinanceExchange  # Alternative

# Swapper MEXC → Binance
configure_container(exchange=BinanceExchange())

# Ou via factory
from src.tools.market import create_exchange
exchange = create_exchange("binance")  # Au lieu de "mexc"
```

### Comment ajouter un nouveau tool

```python
from src.interfaces import BaseTool, ToolDefinition
from src.tools.registry import register_tool

@register_tool
class MyNewTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="my_new_tool",
            description="Description pour le LLM",
            parameters={"type": "object", "properties": {}},
            category="observer",  # ou "reflechir" ou "agir"
        )

    async def execute(self, **kwargs) -> Dict:
        return {"result": "ok"}
```

### Comment ajouter une règle de risque

```python
from src.tools.risk import RiskRule, RiskContext, RiskCheckResult

class MyCustomRule(RiskRule):
    @property
    def name(self) -> str:
        return "my_custom_rule"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        if ctx.market.get("my_condition"):
            return RiskCheckResult(approved=False, reason="My rejection reason")
        return RiskCheckResult(approved=True)

# Usage
risk_manager.add_rule(MyCustomRule())
```

### Tâches de Modularité par Phase

| Phase | Tâche | Description |
|-------|-------|-------------|
| 0 | T0.6 | Créer les interfaces ABC et le container IoC |
| 1 | T1.6 | Implémenter MEXCExchange/PaperExchange avec BaseExchange |
| 2 | T2.6 | Implémenter les providers avec les interfaces |
| 4 | T4.6 | Système de plugins pour les tools |
| 5 | T5.6 | Pattern Chain of Responsibility pour les règles de risque |

---

## Limites de Risque (Hard-coded)

| Limite | Valeur | Description |
|--------|--------|-------------|
| max_order_usd | $20 | Maximum par ordre |
| max_equity_pct | 5% | Maximum % du portefeuille par ordre |
| max_daily_trades | 50 | Nombre max de trades/jour |
| max_daily_loss_usd | $50 | Perte max avant halt |
| max_open_positions | 5 | Positions simultanées max |
| max_spread_pct | 2% | Spread max acceptable |

---

## Checklist de Livraison

### MVP (Minimum Viable Product)
- [ ] Bot démarre sans erreur
- [ ] Mode paper trading fonctionnel
- [ ] Boucle Observe→Think→Act complète
- [ ] Risk manager bloque les ordres dangereux
- [ ] UI affiche les pensées et actions
- [ ] Logs dans SQLite

### Production Ready
- [ ] 50+ trades paper réussis
- [ ] Win rate > 40%
- [ ] Tests coverage > 80%
- [ ] Documentation complète
- [ ] Docker déployable
- [ ] 24h sans crash

---

## Instructions pour l'Agent de Coding

### Règles Générales
1. **Ordre d'exécution**: Suivre les phases dans l'ordre (0→1→2→3→4→5→6)
2. **Une tâche à la fois**: Compléter chaque tâche avant de passer à la suivante
3. **Marquer le statut**: Mettre à jour ce fichier quand une tâche est complétée
4. **Tests**: Écrire des tests pour chaque module
5. **Commits**: Commiter après chaque tâche complétée
6. **Sécurité**: Ne jamais contourner les limites de risque

### Bot 100% AUTONOME
Le bot doit être **100% autonome** - il DÉCIDE et AGIT lui-même :
- Pas de "je recommande" ou "je suggère"
- Le LLM appelle `place_order()` directement quand il veut trader
- La couche risk ajuste ou rejette si nécessaire
- Explication du raisonnement AVANT chaque action

### Spécificités MEXC
- Frais: 0% maker / 0.01% taker - optimiser pour ordres limite
- API plus stricte sur rate limits - ajouter délais entre appels
- Pas de passphrase (contrairement à OKX) - juste API key + secret
- Surveiller les nouveaux listings - c'est la spécialité de MEXC

### System Prompt du LLM
Le bot doit recevoir ce type d'instruction :
> "Tu trades sur MEXC. Profite des frais extrêmement bas (0% maker) pour capturer des mouvements de prix plus petits (scalping) si la tendance est incertaine. Surveille les nouveaux listings récents car c'est la spécialité de cet exchange."

---

## Contact & Support

- **Repository**: https://github.com/elzuzu/OtterTrend
- **Issues**: Pour les bugs et suggestions
- **Spec originale**: CHATGPT.md

---

*Dernière mise à jour: 2025-12-03*
