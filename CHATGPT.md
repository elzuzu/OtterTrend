# Spécification Technique : Projet "Gemini-CLI-Bot SocialFi Autonome"

## 1\. Objectif du Projet

Transformer le repository `google-gemini/gemini-cli` (v0.20) en un bot de trading autonome.

  * **Interface :** Conserver le front-end CLI (basé sur la librairie `Rich`) pour visualiser le "Chain of Thought" et les appels d'outils en temps réel.
  * **Cerveau :** Remplacer l'API Gemini par **Groq** (Free Tier) via le SDK compatible OpenAI.
  * **Marché :** Connexion **OKX** via `ccxt`.
  * **Stratégie :** Analyse de Trend (Google Trends + SocialFi) + Exécution autonome.
  * **Persistance :** `sqlite3`.

## 2\. Architecture Technique

### A. Stack

  * **Langage :** Python 3.10+
  * **LLM Provider :** Groq Cloud API.
  * **Modèle cible :** `llama-3.3-70b-versatile` (Meilleur rapport raisonnement/vitesse/coût pour le free tier).
  * **Trading Lib :** `ccxt` (async si possible pour ne pas bloquer l'UI).
  * **Trend Lib :** `pytrends` (Google Trends), `requests` (API agrégateurs).
  * **CLI UI :** `rich` (déjà présent dans gemini-cli).

### B. Structure du Fork

On ne garde que l'enveloppe de `gemini-cli`. Le cœur (`core`) doit être modifié.

```text
gemini-cli-bot/
├── main.py                # Point d'entrée modifié (boucle infinie)
├── src/
│   ├── client/
│   │   └── groq_adapter.py # Adaptateur simulant l'interface Gemini mais appelant Groq
│   ├── bot/
│   │   ├── brain.py       # Gestion du contexte et des décisions
│   │   ├── memory.py      # Gestion SQLite3
│   │   └── loop.py        # Orchestrateur (Cycle : Observe -> Think -> Act)
│   ├── tools/             # Les "bras" du bot
│   │   ├── market.py      # CCXT Wrapper (OKX)
│   │   ├── trends.py      # Google Trends & Social Scrapers
│   │   ├── risk.py        # Garde-fous (Hard-coded)
│   │   └── analytics.py   # Mini-ML (Volatility, Sentiment simple)
│   └── ui/
│       └── renderer.py    # Affiche le streaming du "Thought" et les Logs
```

## 3\. Spécifications Détaillées des Modules

### I. Base de Données (SQLite3)

Créer un fichier `bot_data.db` avec le schéma suivant :

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    side TEXT, -- 'buy' or 'sell'
    amount REAL,
    price REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    pnl REAL,
    status TEXT -- 'open', 'closed'
);

CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    level TEXT, -- 'INFO', 'DECISION', 'ERROR'
    message TEXT,
    context_snapshot JSON -- Snapshot des données au moment de la décision
);

CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

### II. L'Adaptateur Groq (`src/client/groq_adapter.py`)

L'application d'origine attend un client Gemini. Il faut créer une classe qui imite cette structure mais redirige vers Groq.

  * **Input :** Messages standard.
  * **System Prompt :** Doit être injecté ici pour forcer le comportement "Bot de Trading".
  * **Streaming :** Doit convertir les chunks de réponse Groq pour qu'ils soient affichables par le renderer `rich` de la CLI existante.
  * **Tool Calling :** Groq supporte le function calling. Il faut mapper les outils définis dans `src/tools/` au format JSON schema attendu par Groq.

### III. Les Outils ("Tools") pour le LLM

Le LLM aura accès aux fonctions suivantes (via function calling) :

**1. `trends_tools`**

  * `get_google_trends(keywords: list)`: Retourne le score d'intérêt (0-100) via `pytrends`.
  * `get_social_sentiment(symbol: str)`: (Version MVP) Scrape les titres récents de news crypto aggregator pour extraire un sentiment basique (-1 à 1).

**2. `market_tools` (via CCXT/OKX)**

  * `get_market_price(symbol: str)`: Prix actuel.
  * `get_account_balance()`: Solde disponible (USDT).
  * `get_open_positions()`: Positions actuelles.

**3. `execution_tools` (Autonome)**

  * `place_order(symbol: str, side: str, amount: float, type: str = 'market')`:
      * **IMPORTANT :** Cette fonction doit appeler `risk_manager.validate_order(...)` AVANT d'envoyer à OKX.
  * `close_position(symbol: str)`: Ferme tout sur un symbole.

### IV. Le Cerveau & Le Prompt Système

C'est ici que la magie opère. Le `system_prompt` envoyé à Groq (Llama 3.3) doit être strict :

> "Tu es un Bot de Trading Autonome spécialisé dans les narratifs SocialFi et Crypto Trends.
> Ton objectif est un ROI journalier \> 1%.
> Tu as un contrôle total sur le compte OKX via tes outils.
>
> RÈGLES DE CONDUITE :
>
> 1.  **Thinking Process :** Avant chaque action, tu dois expliquer ton raisonnement dans des balises ).

  - Les appels d'outils en jaune.
  - Les confirmations d'ordres en vert.
  - Les erreurs en rouge.

**Livrables attendus :**

  - Le fichier `main.py` modifié.
  - Le module `groq_adapter.py`.
  - Le module `tools.py`.
  - Le fichier `requirements.txt` (doit inclure : `groq`, `ccxt`, `rich`, `pytrends`, `python-dotenv`).

Commence par l'architecture des fichiers, puis fournis le code pour l'adaptateur Groq et la boucle principale.

````

---

### 5. Mise en garde & Next Step
Ce système sera **autonome**. S'il y a un bug dans le LLM (hallucination sur un prix), il peut vider le compte.
**Impératif :** Dans le fichier `tools/risk.py`, tu dois coder une limite "Hard" :
```python
def check_risk(amount_usd, balance_usd):
    if amount_usd > 20.0: # Limite stricte pour tests
        return False, "Montant trop élevé pour le test"
    if amount_usd > (balance_usd * 0.05):
        return False, "Max 5% equity rule"
    return True, "OK"
````

C’est très cohérent comme spec. Tu as déjà 80 % de l’architecture mentale, il manque juste un peu de “mise en code” et quelques détails d’interface.

Je te propose :

1. Arborescence de fichiers légèrement raffinée.
2. Code pour `src/client/groq_adapter.py`.
3. Code pour la boucle principale (`main.py` + squelette `src/bot/loop.py`).

---

## 1. Arborescence proposée

```text
gemini-cli-bot/
├── main.py
├── requirements.txt
├── src/
│   ├── client/
│   │   └── groq_adapter.py
│   ├── bot/
│   │   ├── brain.py        # (logique 'policy' LLM, éventuels helpers)
│   │   ├── memory.py       # SQLite3 (trades/logs/config)
│   │   └── loop.py         # Boucle Observe -> Think -> Act
│   ├── tools/
│   │   ├── market.py       # CCXT/OKX wrapper
│   │   ├── trends.py       # Google Trends + news/sentiment simple
│   │   ├── risk.py         # Garde-fous (hard rules)
│   │   └── analytics.py    # Mini-ML / stats de base
│   └── ui/
│       └── renderer.py     # Affichage Rich (thoughts, logs, ordres)
└── bot_data.db             # SQLite (créé au runtime)
```

`requirements.txt` minimum (à ajuster) :

```txt
groq
ccxt
rich
pytrends
python-dotenv
sqlite-utils
```

---

## 2. Adaptateur Groq – `src/client/groq_adapter.py`

Objectif : interface simple qui :

* accepte `messages` au format OpenAI,
* gère un `system_prompt`,
* supporte le **function calling**,
* stream les réponses (texte + éventuels tool_calls) pour le renderer.

```python
# src/client/groq_adapter.py

import os
from typing import Any, Dict, Generator, Iterable, List, Optional

from groq import Groq


class GroqAdapter:
    """
    Adaptateur simple pour utiliser Groq (Llama 3.3) en mode OpenAI-like
    avec support du streaming et du function calling.

    L'interface est volontairement minimale :
      - stream_chat() -> génère des événements pour le renderer et la boucle du bot
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY manquant dans l'environnement.")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []

    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Injecte le system prompt si nécessaire.
        On suppose que `messages` est une liste de dicts au format:
        {"role": "user"|"assistant"|"system"|"tool", "content": "...", ...}
        """
        msgs: List[Dict[str, Any]] = []
        has_system = any(m["role"] == "system" for m in messages)
        if self.system_prompt and not has_system:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(messages)
        return msgs

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Lance un chat en streaming.
        Génère des events de forme:

        - {"type": "token", "content": "texte..."}
        - {"type": "tool_call", "name": "...", "arguments": {...}}
        - {"type": "done"} quand la génération est terminée.
        """

        payload_tools = tools if tools is not None else self.tools

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            tools=payload_tools if payload_tools else None,
            tool_choice=tool_choice if payload_tools else None,
        )

        # Suivi des tool_calls en cours
        tool_call_buffers: Dict[str, Dict[str, Any]] = {}

        for chunk in completion:
            choice = chunk.choices[0]

            # 1) texte "normal"
            delta = choice.delta
            if delta.content:
                yield {
                    "type": "token",
                    "content": delta.content,
                }

            # 2) function calling (tool_calls)
            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    call_id = tool_delta.id
                    if call_id not in tool_call_buffers:
                        tool_call_buffers[call_id] = {
                            "id": call_id,
                            "name": "",
                            "arguments": "",
                        }

                    buf = tool_call_buffers[call_id]
                    if tool_delta.function:
                        if tool_delta.function.name:
                            buf["name"] += tool_delta.function.name
                        if tool_delta.function.arguments:
                            buf["arguments"] += tool_delta.function.arguments

            # 3) fin de génération -> on flush les tool_calls
            if choice.finish_reason in ("stop", "tool_calls"):
                for call in tool_call_buffers.values():
                    # Essayer de parser les arguments JSON
                    args = call["arguments"]
                    try:
                        import json

                        parsed_args = json.loads(args) if args else {}
                    except Exception:
                        parsed_args = {"_raw": args}

                    yield {
                        "type": "tool_call",
                        "name": call["name"],
                        "arguments": parsed_args,
                        "id": call["id"],
                    }

                tool_call_buffers.clear()
                yield {"type": "done"}
                break
```

Tu peux ensuite enregistrer la définition des tools (au format JSON schema OpenAI) dans une constante et la passer au constructeur du `GroqAdapter` ou à `stream_chat`.

---

## 3. Boucle principale – `main.py` + `src/bot/loop.py`

### 3.1. Boucle haute-niveau – `main.py`

On garde une CLI simple : initialisation + lancement de la boucle du bot.
La partie UI Rich sera ensuite branchée via `renderer.py`.

```python
# main.py

import asyncio
import os
from dotenv import load_dotenv

from src.client.groq_adapter import GroqAdapter
from src.bot.loop import TradingBotLoop

# Prompt système de base (à affiner)
SYSTEM_PROMPT = """
Tu es un Bot de Trading Autonome spécialisé dans les narratifs SocialFi et Crypto Trends.
Tu contrôles des outils de marché, de tendances et d'exécution (OKX via CCXT).

Objectif principal: maximiser le ROI quotidien cible de >1% sur le long terme,
tout en respectant strictement les garde-fous de risque codés dans les outils.

Tu dois:
- Observer les données marché, tendances et SocialFi via les outils.
- Décider quand ouvrir, ajuster ou fermer des positions.
- Appeler directement les outils d'exécution (place_order, close_position, etc.).
- Expliquer ton raisonnement sous forme de texte continu avant chaque action
  (sans balises techniques, juste du texte clair).
- Limiter la verbosité: aller à l'essentiel pour rester lisible dans un terminal.
"""


async def main() -> None:
    load_dotenv()

    groq_client = GroqAdapter(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        system_prompt=SYSTEM_PROMPT,
    )

    bot_loop = TradingBotLoop(groq_client=groq_client)

    # Boucle infinie du bot autonome
    await bot_loop.run_forever(interval_seconds=60)


if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3.2. Squelette de la boucle – `src/bot/loop.py`

Ici on implémente le fameux cycle **Observe → Think → Act**, sans encore détailler tous les outils.
C’est un squelette qui te donne la structure à brancher dans `market.py`, `trends.py`, etc.

````python
# src/bot/loop.py

import asyncio
import datetime as dt
from typing import Any, Dict, List

from src.client.groq_adapter import GroqAdapter
from src.bot.memory import BotMemory
from src.tools import market, trends, risk
# from src.ui.renderer import Renderer  # à implémenter ensuite


class TradingBotLoop:
    """
    Orchestrateur du bot autonome.

    - Observe: récupère snapshot marché + tendances + portefeuille
    - Think: envoie ce snapshot au LLM (GroqAdapter) et récupère une décision
    - Act: exécute les ordres (via tools) après risk-check
    """

    def __init__(self, groq_client: GroqAdapter) -> None:
        self.groq = groq_client
        self.memory = BotMemory("bot_data.db")
        # self.renderer = Renderer()  # UI Rich (optionnel à ce stade)

    async def run_forever(self, interval_seconds: int = 60) -> None:
        while True:
            loop_started_at = dt.datetime.utcnow()
            try:
                snapshot = await self._observe()
                actions = await self._think(snapshot)
                await self._act(snapshot, actions)
            except Exception as exc:  # garde-fou global
                self.memory.log_error(f"Erreur loop: {exc}", context={})
            finally:
                elapsed = (dt.datetime.utcnow() - loop_started_at).total_seconds()
                sleep_for = max(0, interval_seconds - elapsed)
                await asyncio.sleep(sleep_for)

    async def _observe(self) -> Dict[str, Any]:
        """
        Récupère les données nécessaires pour le LLM :
          - marché (prix, positions)
          - tendances (Google Trends, social)
          - éventuels signaux analytics
        """
        # Ces fonctions seront async dans tools/market.py et tools/trends.py
        market_state = await market.get_market_snapshot()
        portfolio_state = await market.get_portfolio_state()
        trend_state = await trends.get_trend_snapshot()

        snapshot: Dict[str, Any] = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "market": market_state,
            "portfolio": portfolio_state,
            "trends": trend_state,
        }

        return snapshot

    async def _think(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Envoie le snapshot au LLM via GroqAdapter, récupère un plan d'actions.
        On travaille avec un seul message 'user' qui décrit le state.
        Le LLM renvoie du texte + éventuellement des appels d'outils.

        Ici, pour garder le contrôle, on lui demande explicitement de sortir
        une structure JSON d'actions à la fin.
        """
        import json

        user_content = (
            "Voici le snapshot d'état actuel (marché, portefeuille, tendances). "
            "Analyse-le et décide des actions à réaliser maintenant.\n\n"
            f"SNAPSHOT_JSON:\n```json\n{json.dumps(snapshot)}\n```"
            "\n\n"
            "Retourne à la fin un objet JSON avec une clé 'actions', par exemple:\n"
            "{ \"actions\": ["
            " {\"type\": \"OPEN\", \"symbol\": \"BTC/USDT\", \"side\": \"buy\", "
            "\"size_pct_equity\": 0.02},"
            " {\"type\": \"CLOSE\", \"symbol\": \"ETH/USDT\"}"
            "] }."
        )

        messages = [
            {"role": "user", "content": user_content},
        ]

        buffered_text = ""
        actions: List[Dict[str, Any]] = []

        # On utilise le streaming pour alimenter l'UI (renderer), mais ici on
        # ne fait qu'accumuler le texte et extraire le JSON final.
        async for event in self._stream_llm(messages):
            if event["type"] == "token":
                token = event["content"]
                buffered_text += token
                # if self.renderer:
                #     self.renderer.write_thought(token)
            elif event["type"] == "done":
                # Fin de génération -> extraire le JSON
                extracted = self._extract_actions_from_text(buffered_text)
                if extracted:
                    actions = extracted
                break
            elif event["type"] == "tool_call":
                # Dans une version avancée, tu peux laisser le LLM appeler
                # directement les tools ici, mais pour le MVP, on reste simple.
                pass

        # Log du reasoning brut
        self.memory.log_decision(
            message="LLM decision",
            raw_output=buffered_text,
            context_snapshot=snapshot,
        )

        return actions

    async def _stream_llm(self, messages: List[Dict[str, Any]]):
        """
        Wrapper async pour le générateur sync de GroqAdapter.
        """
        loop = asyncio.get_running_loop()

        def _run_sync():
            return list(self.groq.stream_chat(messages=messages))

        # On exécute le générateur entier dans un thread pour rester async-friendly.
        events = await loop.run_in_executor(None, _run_sync)
        for ev in events:
            yield ev

    def _extract_actions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait l'objet JSON { "actions": [...] } depuis la sortie texte du LLM.
        On cherche le dernier bloc JSON valide.
        """
        import json
        import re

        # Stratégie simple : chercher le dernier bloc {...} qui commence par "actions"
        candidates = re.findall(r"\{[^{}]*\"actions\"[^{}]*\}", text, flags=re.DOTALL)
        if not candidates:
            return []

        for raw in reversed(candidates):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "actions" in obj:
                    actions = obj["actions"]
                    if isinstance(actions, list):
                        return actions
            except Exception:
                continue

        return []

    async def _act(
        self,
        snapshot: Dict[str, Any],
        actions: List[Dict[str, Any]],
    ) -> None:
        """
        Exécute les actions proposées, en respectant les garde-fous.
        actions = liste de dicts:
        {
          "type": "OPEN" | "CLOSE" | "ADJUST",
          "symbol": "BTC/USDT",
          "side": "buy"/"sell",
          "size_pct_equity": 0.02,
          ...
        }
        """
        if not actions:
            return

        balance = snapshot["portfolio"].get("balance_usdt", 0.0)

        for action in actions:
            atype = action.get("type", "").upper()
            symbol = action.get("symbol")
            if not symbol:
                continue

            if atype == "OPEN":
                side = action.get("side", "buy").lower()
                size_pct = float(action.get("size_pct_equity", 0.01))
                amount_usd = balance * size_pct

                ok, reason = risk.check_risk(amount_usd=amount_usd, balance_usd=balance)
                if not ok:
                    self.memory.log_info(
                        f"Ordre OPEN refusé par le risk manager: {reason}",
                        context={"action": action},
                    )
                    continue

                # Conversion en quantité de coin (approx prix spot)
                price = await market.get_market_price(symbol)
                if price <= 0:
                    self.memory.log_error(
                        "Prix <= 0, impossible d'ouvrir la position",
                        context={"symbol": symbol},
                    )
                    continue

                amount = amount_usd / price

                order = await market.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    order_type="market",
                )
                self.memory.log_trade_open(order, snapshot, action)

            elif atype == "CLOSE":
                order = await market.close_position(symbol)
                if order:
                    self.memory.log_trade_close(order, snapshot, action)

            # Tu complèteras ici les autres types: "ADJUST", "TAKE_PROFIT", etc.
````

---

### 3.3. Squelette ultra simple de `memory.py` et `risk.py` (optionnel mais utile)

`src/bot/memory.py` : juste pour montrer l’intention.

```python
# src/bot/memory.py

import json
import sqlite3
from typing import Any, Dict, Optional


class BotMemory:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pnl REAL,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT,
                context_snapshot TEXT
            );

            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self.conn.commit()

    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO logs (level, message, context_snapshot) VALUES (?, ?, ?)",
            (level, message, json.dumps(context or {})),
        )
        self.conn.commit()

    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.log("INFO", message, context)

    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.log("ERROR", message, context)

    def log_decision(
        self,
        message: str,
        raw_output: str,
        context_snapshot: Dict[str, Any],
    ):
        ctx = dict(context_snapshot)
        ctx["_raw_llm_output"] = raw_output
        self.log("DECISION", message, ctx)

    def log_trade_open(self, order: Dict[str, Any], snapshot, action):
        self.log("TRADE_OPEN", f"Ouverture trade {order}", {"order": order, "action": action})

    def log_trade_close(self, order: Dict[str, Any], snapshot, action):
        self.log("TRADE_CLOSE", f"Clôture trade {order}", {"order": order, "action": action})
```

`src/tools/risk.py` avec tes contraintes “hard” :

```python
# src/tools/risk.py

def check_risk(amount_usd: float, balance_usd: float):
    if amount_usd > 20.0:
        return False, "Montant trop élevé pour le test"
    if balance_usd <= 0:
        return False, "Balance nulle"
    if amount_usd > (balance_usd * 0.05):
        return False, "Max 5% equity rule"
    return True, "OK"
```

---

OK, on déroule tout proprement.

Je te donne :

1. Roadmap globale (phases à suivre).
2. Spéc détaillée de chaque module manquant + squelettes de code.
3. Intégration du function calling (tools) côté LLM.
4. Check-list avant de le laisser trader en vrai.

---

## 1. Roadmap globale

Ordre logique pour un dev :

1. **Phase 0 – Base existante**

   * Mettre en place le repo, copier la structure, ajouter `groq_adapter.py`, `loop.py`, `memory.py`, `risk.py`, `main.py` (ce qu’on a déjà).

2. **Phase 1 – Marché & portefeuille (OKX via CCXT)**

   * `src/tools/market.py`
   * Variables d’env : clés API OKX, mode testnet/paper.

3. **Phase 2 – Trends & Social (MVP)**

   * `src/tools/trends.py` (pytrends + quelques agrégateurs simples / RSS).
   * `src/tools/analytics.py` (regime/volatilité basique).

4. **Phase 3 – UI Rich / Renderer**

   * `src/ui/renderer.py` pour voir :

     * les “thoughts” du LLM,
     * les décisions / ordres,
     * PnL minimal.

5. **Phase 4 – Function calling**

   * Définir les schemas JSON des tools pour Groq.
   * Router les `tool_call` → fonctions Python (market/trends/etc.).
   * Boucle LLM multi-tours : think → tool → think → actions.

6. **Phase 5 – Mode paper trading & garde-fous**

   * Flag `PAPER_TRADING=true` dans `.env`.
   * Market wrapper qui simule l’exécution si paper.
   * Limites de taille et de pertes journalières.

7. **Phase 6 – Tests & durcissement**

   * Tests unitaires simples sur tools.
   * Dry-run 24–72 h en paper.
   * Ensuite seulement petit capital réel.

---

## 2. `src/tools/market.py` – OKX / CCXT

Objectif : wrapper unique OKX (spot pour commencer), async-friendly, avec un mode paper.

### 2.1. Config .env

```env
GROQ_API_KEY=...
OKX_API_KEY=...
OKX_API_SECRET=...
OKX_API_PASSWORD=...   # passphrase OKX
OKX_TESTNET=true        # ou false
PAPER_TRADING=true      # pour simuler
BASE_CURRENCY=USDT
```

### 2.2. Code

```python
# src/tools/market.py

import os
import asyncio
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

BASE_CCY = os.getenv("BASE_CURRENCY", "USDT")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

_exchange: Optional[ccxt.okx] = None

# État simple pour le mode paper
_paper_balance_usdt: float = 1000.0
_paper_positions: Dict[str, Dict[str, Any]] = {}


def _get_okx_kwargs() -> Dict[str, Any]:
    testnet = os.getenv("OKX_TESTNET", "true").lower() == "true"
    return {
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSWORD"),
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
        "test": testnet,
    }


def get_exchange() -> ccxt.okx:
    global _exchange
    if _exchange is None:
        kwargs = _get_okx_kwargs()
        _exchange = ccxt.okx(kwargs)
    return _exchange


async def get_market_price(symbol: str) -> float:
    """
    symbol ex: 'BTC/USDT'
    """
    if PAPER_TRADING:
        # en paper, on peut simuler un prix stable ou utiliser le vrai prix
        ex = get_exchange()
        ticker = await ex.fetch_ticker(symbol)
        return float(ticker["last"])

    ex = get_exchange()
    ticker = await ex.fetch_ticker(symbol)
    return float(ticker["last"])


async def get_market_snapshot(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    ex = get_exchange()

    if symbols is None:
        # pour un MVP: quelques paires
        symbols = ["BTC/USDT", "ETH/USDT"]

    tickers = await ex.fetch_tickers(symbols)
    out = {}
    for sym, data in tickers.items():
        out[sym] = {
            "last": float(data["last"]),
            "volume": float(data["baseVolume"] or 0),
            "quoteVolume": float(data["quoteVolume"] or 0),
        }
    return out


async def get_portfolio_state() -> Dict[str, Any]:
    global _paper_balance_usdt, _paper_positions

    if PAPER_TRADING:
        return {
            "mode": "paper",
            "balance_usdt": _paper_balance_usdt,
            "positions": _paper_positions,
        }

    ex = get_exchange()
    balance = await ex.fetch_balance()
    total = balance.get("total", {})
    free = balance.get("free", {})

    base_bal = float(total.get(BASE_CCY, 0))
    # positions spot: estimation simple
    positions: Dict[str, Any] = {}

    # Option simple : parcourt les balances non-USD et calcule en USDT
    for asset, qty in total.items():
        if asset.upper() == BASE_CCY.upper():
            continue
        if qty is None or qty == 0:
            continue
        sym = f"{asset}/{BASE_CCY}"
        try:
            price = await get_market_price(sym)
        except Exception:
            price = 0
        positions[sym] = {
            "amount": float(qty),
            "value_usdt": float(qty) * price,
        }

    return {
        "mode": "live",
        "balance_usdt": base_bal,
        "positions": positions,
        "raw": {"total": total, "free": free},
    }


async def place_order(
    symbol: str,
    side: str,
    amount: float,
    order_type: str = "market",
) -> Dict[str, Any]:
    """
    Wrapper d'exécution.
    En mode PAPER_TRADING, on simule.
    """
    global _paper_balance_usdt, _paper_positions

    side = side.lower()
    if PAPER_TRADING:
        price = await get_market_price(symbol)
        cost = amount * price

        if side == "buy":
            if cost > _paper_balance_usdt:
                raise ValueError("Balance insuffisante PAPER")
            _paper_balance_usdt -= cost
            pos = _paper_positions.get(symbol, {"amount": 0.0, "avg_price": 0.0})
            new_amount = pos["amount"] + amount
            if new_amount > 0:
                new_avg = (pos["amount"] * pos["avg_price"] + amount * price) / new_amount
            else:
                new_avg = price
            _paper_positions[symbol] = {"amount": new_amount, "avg_price": new_avg}
        else:  # sell
            pos = _paper_positions.get(symbol, {"amount": 0.0, "avg_price": 0.0})
            if amount > pos["amount"]:
                amount = pos["amount"]
            revenue = amount * price
            _paper_balance_usdt += revenue
            new_amount = pos["amount"] - amount
            if new_amount <= 0:
                _paper_positions.pop(symbol, None)
            else:
                _paper_positions[symbol]["amount"] = new_amount

        return {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "mode": "paper",
        }

    ex = get_exchange()
    if order_type == "market":
        order = await ex.create_order(symbol, "market", side, amount)
    else:
        # à enrichir plus tard
        order = await ex.create_order(symbol, order_type, side, amount)

    return order


async def close_position(symbol: str) -> Optional[Dict[str, Any]]:
    global _paper_balance_usdt, _paper_positions

    if PAPER_TRADING:
        pos = _paper_positions.get(symbol)
        if not pos or pos["amount"] <= 0:
            return None
        price = await get_market_price(symbol)
        amount = pos["amount"]
        revenue = amount * price
        _paper_balance_usdt += revenue
        _paper_positions.pop(symbol, None)
        return {
            "symbol": symbol,
            "side": "sell",
            "amount": amount,
            "price": price,
            "mode": "paper_close",
        }

    # spot: on considère qu'on vend simplement tout ce qu'on a
    ex = get_exchange()
    balance = await ex.fetch_balance()
    base = symbol.split("/")[0]
    qty = float(balance["free"].get(base, 0))
    if qty <= 0:
        return None
    order = await ex.create_order(symbol, "market", "sell", qty)
    return order
```

---

## 3. `src/tools/trends.py` – Google Trends + news crypto

MVP = Google Trends + simple sentiment sur titres de news.

```python
# src/tools/trends.py

import asyncio
import datetime as dt
from typing import Any, Dict, List

from pytrends.request import TrendReq
import requests

_pytrends = TrendReq(hl="en-US", tz=0)


async def get_google_interest(keywords: List[str], timeframe: str = "now 7-d") -> Dict[str, Any]:
    """
    Retourne {keyword: {score_moyen, last, series}} pour chaque mot-clé.
    """
    loop = asyncio.get_running_loop()

    def _fetch():
        _pytrends.build_payload(keywords, timeframe=timeframe, geo="")
        data = _pytrends.interest_over_time()
        out = {}
        if data.empty:
            return out
        for kw in keywords:
            series = data[kw].tolist()
            last = series[-1]
            avg = sum(series) / len(series)
            out[kw] = {
                "avg": avg,
                "last": last,
                "series": series[-20:],  # derniers points
            }
        return out

    res = await loop.run_in_executor(None, _fetch)
    return res


def _simple_sentiment_from_title(title: str) -> float:
    """
    Heuristique débile mais suffisante pour un MVP.
    """
    title_low = title.lower()
    if any(w in title_low for w in ["surge", "rally", "up", "all-time high", "soars"]):
        return 0.8
    if any(w in title_low for w in ["dump", "crash", "down", "liquidation"]):
        return -0.8
    return 0.0


async def get_crypto_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    MVP: interroger un agrégateur type Coindesk/Cointelegraph RSS (ou un endpoint maison),
    parser quelques titres, et en déduire un sentiment moyen.
    Ici, on laisse un pseudo-code, le dev branchera une vraie source.
    """
    # exemple avec un placeholder simple (à adapter)
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    resp = requests.get(url, timeout=5)
    data = resp.json()
    titles = []
    for item in data.get("Data", [])[:30]:
        t = item.get("title", "")
        if symbol.upper().split("/")[0] in t.upper():
            titles.append(t)

    if not titles:
        return {"symbol": symbol, "sentiment": 0.0, "titles": []}

    scores = [_simple_sentiment_from_title(t) for t in titles]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"symbol": symbol, "sentiment": avg, "titles": titles[:5]}


async def get_trend_snapshot() -> Dict[str, Any]:
    """
    Snapshot global pour le LLM:
    - quelques mots-clés génériques,
    - éventuellement quelques tokens SocialFi suivis.
    """
    keywords = ["socialfi", "crypto airdrop", "memecoin"]
    g = await get_google_interest(keywords)

    # Symboles SocialFi à suivre : placeholder, à configurer
    socialfi_symbols = ["CYBER/USDT", "DEGEN/USDT"]
    news_sentiments = {}
    for sym in socialfi_symbols:
        try:
            news_sentiments[sym] = await get_crypto_news_sentiment(sym)
        except Exception:
            continue

    return {
        "google_trends": g,
        "news_sentiment": news_sentiments,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
```

---

## 4. `src/tools/analytics.py` – mini-ML basique

Juste de quoi donner quelques signaux au LLM.

```python
# src/tools/analytics.py

from typing import Dict, List
import numpy as np


def compute_return_series(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    return [ (prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices)) ]


def detect_regime(prices: List[float]) -> str:
    """
    Regime basique:
    - tendance haussière: drift > 0
    - baissière: drift < 0
    - range: drift ~0 et vola basse
    - chaos: vola très haute
    """
    rets = compute_return_series(prices)
    if not rets:
        return "UNKNOWN"
    arr = np.array(rets)
    drift = arr.mean()
    vol = arr.std()

    if vol > 0.08:
        return "CHAOS"
    if drift > 0.002:
        return "TREND_UP"
    if drift < -0.002:
        return "TREND_DOWN"
    return "RANGE"


def volatility_score(prices: List[float]) -> float:
    rets = compute_return_series(prices)
    if not rets:
        return 0.0
    return float(np.std(rets))
```

Ensuite, tu peux intégrer ces signaux dans `get_market_snapshot` (ou dans une autre fonction) pour fournir au LLM :

```python
"analytics": {
    "BTC/USDT": {"regime": "TREND_UP", "vol_score": 0.02},
    ...
}
```

---

## 5. `src/ui/renderer.py` – interface Rich simplifiée

Objectif : montrer en temps réel :

* les tokens de pensée du LLM,
* les décisions/actions,
* un petit résumé PnL.

```python
# src/ui/renderer.py

from typing import Optional
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.console import Console

console = Console()


class Renderer:
    def __init__(self) -> None:
        self.thought_buffer = ""
        self.last_actions = []
        self.last_pnl = {}

    def build_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="upper", ratio=3),
            Layout(name="lower", ratio=1),
        )
        layout["upper"].split_row(
            Layout(name="thoughts"),
            Layout(name="actions"),
        )
        return layout

    def _thoughts_panel(self) -> Panel:
        return Panel(self.thought_buffer[-2000:], title="Thoughts")

    def _actions_panel(self) -> Panel:
        table = Table(title="Actions récentes", show_header=True)
        table.add_column("Type")
        table.add_column("Symbol")
        table.add_column("Détails")
        for a in self.last_actions[-10:]:
            table.add_row(a.get("type", ""), a.get("symbol", ""), str(a))
        return Panel(table)

    def _pnl_panel(self) -> Panel:
        table = Table(title="PNL / Portfolio")
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in self.last_pnl.items():
            table.add_row(k, str(v))
        return Panel(table)

    def live_loop(self, bot_loop_coroutine):
        """
        Démarre un Live Rich et exécute le coroutine du bot à côté.
        Le bot appelle les méthodes write_thought / register_action / update_pnl.
        """
        layout = self.build_layout()
        with Live(layout, refresh_per_second=5, console=console):
            import asyncio

            async def _runner():
                await bot_loop_coroutine

            asyncio.create_task(_runner())
            try:
                while True:
                    layout["upper"]["thoughts"].update(self._thoughts_panel())
                    layout["upper"]["actions"].update(self._actions_panel())
                    layout["lower"].update(self._pnl_panel())
                    # refresh automatique via Live
            except KeyboardInterrupt:
                return

    def write_thought(self, token: str):
        self.thought_buffer += token

    def register_action(self, action):
        self.last_actions.append(action)

    def update_pnl(self, pnl_dict):
        self.last_pnl = pnl_dict
```

Dans `TradingBotLoop`, tu peux brancher le renderer :

```python
# dans __init__
self.renderer = Renderer()

# dans _stream_llm
if event["type"] == "token":
    token = event["content"]
    buffered_text += token
    if self.renderer:
        self.renderer.write_thought(token)
```

Et à chaque exécution d’action :

```python
self.renderer.register_action(action)
```

---

## 6. Function calling / Tools pour le LLM

Si tu veux aller au bout de ta spec initiale (tools Groq), tu définis un schema de tools côté `GroqAdapter` ou dans un module `tools/schema.py`.

### 6.1. Exemple de schema tools

```python
# src/tools/schemas.py

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_google_trends",
            "description": "Récupère l'intérêt Google Trends pour une liste de mots-clés",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "timeframe": {
                        "type": "string",
                        "default": "now 7-d",
                    },
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_price",
            "description": "Récupère le prix actuel d'un symbole (spot)",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Passe un ordre marché sur un symbole, après risk-check",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "amount_usd": {"type": "number"},
                },
                "required": ["symbol", "side", "amount_usd"],
            },
        },
    },
    # etc. pour close_position, get_portfolio_state, etc.
]
```

Et tu passes `TOOLS` au `GroqAdapter` :

```python
from src.tools.schemas import TOOLS

groq_client = GroqAdapter(
    api_key=...,
    model="llama-3.3-70b-versatile",
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
```

### 6.2. Router les tool_calls

Dans `TradingBotLoop._think`, tu peux gérer les `tool_call` renvoyés par `GroqAdapter` :

* boucle :

  * LLM répond avec un `tool_call`,
  * tu exécutes la fonction Python correspondante,
  * tu renvoies la réponse à LLM comme message `role="tool"`,
  * tu relances `stream_chat` jusqu’à ce qu’il donne la réponse finale (actions JSON).

Vu la complexité, tu peux dans un premier temps rester avec l’approche actuelle (LLM renvoie seulement un JSON d’actions), et introduire les tools plus tard.

---

## 7. Check-list avant réel

1. **PAPER_TRADING = true**, valider :

   * pas d’erreur de boucle,
   * le bot ouvre/ferme des positions papier,
   * logs dans SQLite OK,
   * UI Rich affiche bien les thoughts et actions.

2. **Test de limites**

   * Forcer le LLM à demander des tailles énormes (en modifiant le prompt), vérifier que `risk.check_risk` bloque bien.

3. **Mode réel, petits montants**

   * `PAPER_TRADING=false`, `OKX_TESTNET=true` (si tu utilises testnet).
   * Puis seulement plus tard : testnet off + limites très basses (hard-coded).

Je vais droit au but.

### 1. Mon avis sur ton pivot OKX → MEXC/Bybit

Pour un bot **petit capital + chasse de trends SocialFi/memecoins + objectif agressif**, ton raisonnement est logique :

* **OKX** = très bon exchange “tier 1”, mais :

  * listings plus lents,
  * frais standards,
  * pas l’endroit le plus agressif pour les microcaps.

* **MEXC** = bon choix pour ton cas :

  * très bon **ratio petits caps / frais / rapidité de listing**,
  * bon terrain de jeu pour un bot “degen trend-hunting”,
  * à traiter comme **exchange d’exécution** (transit), pas comme coffre-fort.

* **Bybit** = plan B propre :

  * meilleure liquidité que MEXC,
  * API très solide,
  * mais un peu moins “degen” en listings que MEXC.

Donc :
Pour ton projet précis (1 %/jour, trends SocialFi/memes, petit capital), **MEXC comme exchange principal est cohérent**, avec la possibilité de rajouter Bybit plus tard comme 2ᵉ venue “liquidité propre”.

---

### 2. Ce qu’il faut **ajouter/ajuster** dans ta spec côté dev

Je te laisse le gros du boulot aux agents de coding, je ne donne que les éléments-clés à intégrer.

---

#### 2.1. Généraliser l’exchange (ne plus hardcoder OKX)

**À ajouter dans la spec :**

* Dans `.env` :

```env
EXCHANGE_ID=mexc        # 'mexc' | 'bybit' | 'okx' ...
EXCHANGE_TESTNET=true   # si applicable
PAPER_TRADING=true
BASE_CURRENCY=USDT
```

* Dans `src/tools/market.py` :

  * Remplacer la factory `ccxt.okx(...)` par quelque chose du genre :

    ```python
    import ccxt.async_support as ccxt
    import os

    EXCHANGE_ID = os.getenv("EXCHANGE_ID", "mexc")

    def get_exchange() -> ccxt.Exchange:
        global _exchange
        if _exchange is None:
            params = {
                "apiKey": os.getenv("EXCHANGE_API_KEY"),
                "secret": os.getenv("EXCHANGE_API_SECRET"),
                "enableRateLimit": True,
            }
            # éventuellement options spécifiques par exchange
            cls = getattr(ccxt, EXCHANGE_ID)
            _exchange = cls(params)
        return _exchange
    ```

  * Documenter dans la spec que :

    * **MEXC** n’a pas de `password/passphrase` comme OKX.
    * `enableRateLimit=True` est important pour MEXC (rate limit plus sensible).

L’agent de coding sait ensuite brancher les bons champs API par exchange.

---

#### 2.2. Spécificités MEXC à mentionner

**Points à inclure pour guider le dev :**

1. **Rate limits**

   * Instruction explicite : ajouter un **petit sleep** dans les boucles intensives (ex : scans de tickers), surtout sur MEXC.
   * Ou s’appuyer sur `exchange.rateLimit` de CCXT pour espacer les appels.

2. **Pairs et notation**

   * Vérifier la notation des symboles (`"TOKEN/USDT"`) et la dispo spot vs futures.
   * Forcer, pour le MVP, le **spot USDT** sur MEXC pour éviter la complexité des contrats dérivés au début.

3. **Slippage / tailles max**

   * Pour un exchange “degen” avec beaucoup de microcaps :

     * limiter **la taille absolue** des ordres (ex : 5–10 USDT par trade au début),
     * toujours passer par l’outil `risk.check_risk` pour écraser les tailles débiles demandées par le LLM.

---

#### 2.3. Trend detection spécifique MEXC

À ajouter dans la spec : un module simple pour exploiter le côté “pépites / nouveaux listings”.

Par exemple dans `src/tools/market.py` ou un module séparé `exchange_intel.py` :

* **Outil 1 : `get_new_listings()`**

  * Utiliser `exchange.fetch_markets()`.
  * Stocker les markets déjà connus en local (SQLite ou mémoire).
  * Détecter les nouveaux symboles apparus depuis le dernier run.
  * Retourner une liste de `{symbol, base, quote, listing_time_estimated}`.

* **Outil 2 : `get_top_gainers_24h()`**

  * Sur MEXC :

    * soit via un endpoint spécifique si dispo,
    * soit en dérivant depuis les tickers (`fetch_tickers` → tri par % change).
  * Exposer au LLM une structure du type :

    ```json
    {
      "exchange": "mexc",
      "top_gainers": [
        {"symbol": "XYZ/USDT", "change_24h": 320.5, "volume_usdt": 123456.0},
        ...
      ]
    }
    ```

Indique dans la spec que le LLM pourra se servir de ces outils pour :

* repérer les **nouveaux listings** MEXC,
* repérer les **gainers SocialFi/memes** à surveiller.

---

#### 2.4. Prompt système à adapter (sans écrire tout le texte)

Donner au dev/agent la consigne suivante :

* Dans `SYSTEM_PROMPT`, faire apparaître explicitement :

  * que l’exchange principal est **MEXC**,
  * que les **frais très bas** autorisent le **scalping micro-mouvements**,
  * que le LLM doit :

    * surveiller **nouveaux listings** et **gainers** via les outils,
    * éviter les ordres trop gros sur les pairs peu liquides,
    * se rappeler qu’il trade sur un exchange “degen” (risque order book + wick violents).

Tu peux formuler pour le dev un bloc à insérer dans le prompt :

> “Tu trades sur MEXC (via CCXT).
> Profite des frais très faibles pour capturer des mouvements de prix même faibles, mais surveille toujours la liquidité et la profondeur du carnet. Utilise les outils `get_new_listings` et `get_top_gainers_24h` pour identifier des opportunités ‘early trend’ (SocialFi, Memecoins) et adapter tes entrées/sorties.”

Ça suffit, le reste du wording, l’agent de coding le gère.

---

#### 2.5. Risk manager : une note spécifique “exchange degen”

Tu as déjà le `check_risk` de base (20 USD max / 5 % d’equity).

Ajoute juste dans la spec :

* Un second garde-fou côté **liquidité** :

  * un outil `estimate_liquidity(symbol)` (volume 24 h, profondeur à ±1 %)
  * une règle du type :

    > si `volume_24h < X` ou `liquidity_±1% < Y`, limiter l’ordre à Z USDT max, ou ne pas trader cette paire.

Le dev n’a pas besoin de tous les détails, juste de ce principe.

---

#### 2.6. Documentation interne pour l’agent de coding

Donne-lui juste ces phrases-clés :

* **Objectif** : optimiser un bot orienté **trends SocialFi/memes à petit capital**, les choix d’exchange doivent refléter ça → **MEXC par défaut**, Bybit ou OKX en option.

* **Abstraction** : ne jamais coder OKX ou MEXC en dur dans la logique métier ; utiliser `EXCHANGE_ID` et des wrappers CCXT.

* **Outils supplémentaires côté exchange** :

  * `get_new_listings()`,
  * `get_top_gainers_24h()`,
  * plus tard éventuellement `get_low_liquidity_pairs()` pour filtrer.

* **Prompt** : clairement informer le LLM qu’il traite avec MEXC, frais bas, petits caps, et qu’il doit exploiter ça avec prudence (fréquence d’ordres élevée mais tailles petites).

---

Si tu veux, je peux te rédiger le petit bloc “Mise à jour de la spéc” que tu colles tel quel dans ton document, avec uniquement les points “Switch OKX → MEXC + outils d’early listing”.
