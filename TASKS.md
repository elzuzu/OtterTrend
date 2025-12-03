# OtterTrend - Plan de Développement

> **Objectif**: Bot de trading autonome SocialFi/Crypto 100% fonctionnel
>
> **Exchange Principal**: MEXC (frais bas, listings rapides)
>
> **Technologie LLM**: Groq (Llama 3.3 70B)
>
> **ROI Cible**: >1% journalier

---

## Vue d'Ensemble des Phases

| Phase | Nom | Description | Priorité | Statut |
|-------|-----|-------------|----------|--------|
| 0 | [Setup & Architecture](/.claude/tasks/phase-0-setup.md) | Structure, config, base de données | CRITIQUE | 🔴 |
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
| T0.1.1 | Créer l'arborescence de fichiers | CRITIQUE | 🔴 |
| T0.1.2 | Créer requirements.txt | CRITIQUE | 🔴 |
| T0.1.3 | Créer .env.example | HAUTE | 🔴 |
| T0.1.4 | Créer .gitignore | HAUTE | 🔴 |
| T0.2.1 | Module de configuration centralisé | HAUTE | 🔴 |
| T0.2.2 | Point d'entrée main.py | CRITIQUE | 🔴 |
| T0.3.1 | Implémenter src/bot/memory.py (SQLite) | CRITIQUE | 🔴 |
| T0.4.1 | Implémenter src/client/groq_adapter.py | CRITIQUE | 🔴 |
| T0.5.1 | Implémenter src/bot/loop.py (squelette) | CRITIQUE | 🔴 |

---

### Phase 1 - Market & Portfolio (MEXC via CCXT)
**Fichier**: [.claude/tasks/phase-1-market.md](/.claude/tasks/phase-1-market.md)

| ID | Tâche | Priorité | Statut |
|----|-------|----------|--------|
| T1.1.1 | Wrapper CCXT multi-exchange | CRITIQUE | 🔴 |
| T1.1.2 | Méthodes de lecture marché | CRITIQUE | 🔴 |
| T1.2.1 | Méthodes de lecture portefeuille | CRITIQUE | 🔴 |
| T1.2.2 | Snapshot marché complet | HAUTE | 🔴 |
| T1.3.1 | Passation d'ordres | CRITIQUE | 🔴 |
| T1.4.1 | Détection nouveaux listings | HAUTE | 🔴 |
| T1.4.2 | Scan top gainers | HAUTE | 🔴 |
| T1.4.3 | Estimation de liquidité | MOYENNE | 🔴 |
| T1.5.1 | Simulateur Paper Trading | CRITIQUE | 🔴 |
| T1.5.2 | Factory exchange client | HAUTE | 🔴 |

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

## Architecture Cible

```
OtterTrend/
├── main.py                          # Point d'entrée
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
│   │   ├── brain.py                 # Logique décisionnelle
│   │   ├── memory.py                # SQLite persistence
│   │   └── loop.py                  # Boucle Observe→Think→Act
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── market.py                # MEXC/CCXT interface
│   │   ├── trends.py                # Google Trends + news
│   │   ├── risk.py                  # Risk manager
│   │   ├── analytics.py             # ML/stats basiques
│   │   ├── schemas.py               # Tools JSON schemas
│   │   └── router.py                # Tool execution router
│   └── ui/
│       ├── __init__.py
│       └── renderer.py              # Rich CLI
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_memory.py
│   ├── test_risk.py
│   ├── test_market.py
│   └── test_integration.py
├── scripts/
│   └── start.py                     # Script de démarrage
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

1. **Ordre d'exécution**: Suivre les phases dans l'ordre (0→1→2→3→4→5→6)
2. **Une tâche à la fois**: Compléter chaque tâche avant de passer à la suivante
3. **Marquer le statut**: Mettre à jour ce fichier quand une tâche est complétée
4. **Tests**: Écrire des tests pour chaque module
5. **Commits**: Commiter après chaque tâche complétée
6. **Sécurité**: Ne jamais contourner les limites de risque

---

## Contact & Support

- **Repository**: https://github.com/elzuzu/OtterTrend
- **Issues**: Pour les bugs et suggestions
- **Spec originale**: CHATGPT.md

---

*Dernière mise à jour: 2025-12-03*
