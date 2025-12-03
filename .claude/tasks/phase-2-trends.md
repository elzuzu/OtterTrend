# Phase 2 - Trends & Social Sentiment

> **Objectif**: Implémenter l'analyse de tendances (Google Trends) et le sentiment social pour identifier les opportunités SocialFi et memecoins.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (structure de base)
- Phase 1 partielle (config, base exchange)

---

## T2.1 - Google Trends Integration

### T2.1.1 - Implémenter le wrapper PyTrends
**Priorité**: CRITIQUE
**Estimation**: Moyenne

Créer `src/tools/trends.py` avec support async pour Google Trends :

```python
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from pytrends.request import TrendReq


class GoogleTrendsClient:
    """
    Client pour récupérer les données Google Trends.
    Note: pytrends est bloquant, on wrappe avec asyncio.
    """

    def __init__(
        self,
        hl: str = "en-US",
        tz: int = 0,
        timeout: tuple = (10, 25),
    ) -> None:
        self._hl = hl
        self._tz = tz
        self._timeout = timeout
        self._pytrends: Optional[TrendReq] = None

    def _get_client(self) -> TrendReq:
        if self._pytrends is None:
            self._pytrends = TrendReq(
                hl=self._hl,
                tz=self._tz,
                timeout=self._timeout,
                retries=2,
                backoff_factor=0.5,
            )
        return self._pytrends

    async def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "now 7-d",
        geo: str = "",
    ) -> Dict[str, Any]:
        """
        Récupère l'intérêt dans le temps pour une liste de mots-clés.

        Args:
            keywords: Liste de mots-clés (max 5)
            timeframe: Période ('now 1-H', 'now 7-d', 'today 1-m', etc.)
            geo: Code pays ('US', 'FR', '' pour global)

        Returns:
            Dict avec score par keyword:
            {
                "keyword": {
                    "current": 75,
                    "avg": 60,
                    "max": 100,
                    "min": 45,
                    "trend": "rising" | "falling" | "stable",
                    "series": [...]
                }
            }
        """
        loop = asyncio.get_running_loop()

        def _fetch():
            client = self._get_client()
            client.build_payload(
                keywords[:5],  # Google limite à 5
                timeframe=timeframe,
                geo=geo,
            )
            df = client.interest_over_time()
            return df

        try:
            df = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            return {"error": str(e), "keywords": keywords}

        if df.empty:
            return {kw: {"current": 0, "avg": 0, "trend": "unknown"} for kw in keywords}

        result = {}
        for kw in keywords:
            if kw not in df.columns:
                result[kw] = {"current": 0, "avg": 0, "trend": "unknown"}
                continue

            series = df[kw].tolist()
            current = series[-1] if series else 0
            avg = sum(series) / len(series) if series else 0
            max_val = max(series) if series else 0
            min_val = min(series) if series else 0

            # Calculer la tendance
            if len(series) >= 2:
                recent_avg = sum(series[-3:]) / min(3, len(series))
                old_avg = sum(series[:3]) / min(3, len(series))
                if recent_avg > old_avg * 1.1:
                    trend = "rising"
                elif recent_avg < old_avg * 0.9:
                    trend = "falling"
                else:
                    trend = "stable"
            else:
                trend = "unknown"

            result[kw] = {
                "current": int(current),
                "avg": round(avg, 1),
                "max": int(max_val),
                "min": int(min_val),
                "trend": trend,
                "series": series[-20:],  # Derniers points
            }

        return result

    async def get_related_queries(
        self,
        keyword: str,
        timeframe: str = "now 7-d",
    ) -> Dict[str, List[str]]:
        """
        Récupère les recherches associées pour un mot-clé.
        Utile pour découvrir des tendances émergentes.
        """
        loop = asyncio.get_running_loop()

        def _fetch():
            client = self._get_client()
            client.build_payload([keyword], timeframe=timeframe)
            return client.related_queries()

        try:
            data = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            return {"error": str(e)}

        result = {"rising": [], "top": []}

        if keyword in data:
            kw_data = data[keyword]
            if "rising" in kw_data and kw_data["rising"] is not None:
                rising_df = kw_data["rising"]
                result["rising"] = rising_df["query"].tolist()[:10]
            if "top" in kw_data and kw_data["top"] is not None:
                top_df = kw_data["top"]
                result["top"] = top_df["query"].tolist()[:10]

        return result

    async def get_trending_searches(
        self,
        geo: str = "united_states",
    ) -> List[str]:
        """
        Récupère les recherches tendances du jour.
        """
        loop = asyncio.get_running_loop()

        def _fetch():
            client = self._get_client()
            df = client.trending_searches(pn=geo)
            return df[0].tolist() if not df.empty else []

        try:
            return await loop.run_in_executor(None, _fetch)
        except Exception:
            return []
```

**Critères de validation**:
- [ ] Wrapper async fonctionnel
- [ ] Gestion des rate limits Google
- [ ] Calcul de tendance (rising/falling/stable)
- [ ] Related queries pour découverte
- [ ] Tests avec données réelles

---

### T2.1.2 - Créer les mots-clés de suivi crypto/SocialFi
**Priorité**: HAUTE
**Estimation**: Simple

Définir les listes de mots-clés à surveiller :

```python
# Dans src/tools/trends.py ou src/config.py

# Mots-clés crypto généraux
CRYPTO_KEYWORDS = [
    "bitcoin",
    "ethereum",
    "solana",
    "crypto airdrop",
    "memecoin",
]

# Mots-clés SocialFi spécifiques
SOCIALFI_KEYWORDS = [
    "socialfi",
    "friend tech",
    "lens protocol",
    "farcaster",
    "social token",
]

# Mots-clés narratifs/trends
NARRATIVE_KEYWORDS = [
    "AI crypto",
    "RWA token",
    "DePIN",
    "restaking",
    "layer 2",
]

# Mapping keyword -> tokens associés (pour corrélation)
KEYWORD_TO_TOKENS = {
    "bitcoin": ["BTC/USDT"],
    "ethereum": ["ETH/USDT"],
    "solana": ["SOL/USDT"],
    "AI crypto": ["FET/USDT", "RNDR/USDT", "AGIX/USDT"],
    "DePIN": ["HNT/USDT", "IOTX/USDT"],
    "restaking": ["EIGEN/USDT", "LDO/USDT"],
}
```

**Critères de validation**:
- [ ] Listes complètes et pertinentes
- [ ] Mapping vers tokens tradeables
- [ ] Facilement extensible

---

## T2.2 - Sentiment Analysis (News)

### T2.2.1 - Implémenter le fetcher de news crypto
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
import requests
from typing import Any, Dict, List
from datetime import datetime, timedelta


class CryptoNewsClient:
    """
    Client pour récupérer les actualités crypto.
    Sources: CryptoCompare, CoinGecko, RSS feeds.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "OtterTrend/1.0",
        })

    async def get_news_cryptocompare(
        self,
        categories: str = "ALL",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les news depuis CryptoCompare API (gratuit).
        """
        loop = asyncio.get_running_loop()

        def _fetch():
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                "lang": "EN",
                "categories": categories,
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("Data", [])[:limit]

        try:
            news = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            return []

        result = []
        for item in news:
            result.append({
                "title": item.get("title", ""),
                "body": item.get("body", "")[:500],  # Tronquer
                "source": item.get("source_info", {}).get("name", "unknown"),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("published_on", 0)
                ).isoformat(),
                "categories": item.get("categories", "").split("|"),
                "tags": item.get("tags", "").split("|"),
            })

        return result

    async def search_news_by_symbol(
        self,
        symbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recherche les news mentionnant un symbole spécifique.
        """
        base_asset = symbol.split("/")[0].upper()
        news = await self.get_news_cryptocompare(limit=100)

        matching = []
        for item in news:
            title_upper = item["title"].upper()
            body_upper = item["body"].upper()
            tags_upper = [t.upper() for t in item["tags"]]

            if (
                base_asset in title_upper
                or base_asset in body_upper
                or base_asset in tags_upper
            ):
                matching.append(item)

            if len(matching) >= limit:
                break

        return matching
```

**Critères de validation**:
- [ ] Fetch depuis API gratuite
- [ ] Filtrage par symbole
- [ ] Données normalisées
- [ ] Gestion timeout et erreurs

---

### T2.2.2 - Implémenter l'analyse de sentiment basique
**Priorité**: HAUTE
**Estimation**: Moyenne

```python
from typing import List, Tuple

# Mots-clés positifs/négatifs pour crypto
POSITIVE_WORDS = [
    "surge", "rally", "soar", "bullish", "breakout", "ath", "all-time high",
    "pump", "moon", "gains", "growth", "adoption", "partnership", "launch",
    "upgrade", "mainnet", "listing", "approval", "institutional",
]

NEGATIVE_WORDS = [
    "crash", "dump", "bearish", "plunge", "liquidation", "hack", "exploit",
    "scam", "rug", "fraud", "ban", "regulation", "sec", "lawsuit",
    "delay", "postpone", "failure", "bug", "vulnerability",
]

NEUTRAL_AMPLIFIERS = ["breaking", "alert", "news", "update"]


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Analyse le sentiment d'un texte.

    Returns:
        (score, label)
        score: -1.0 (très négatif) à +1.0 (très positif)
        label: 'positive', 'negative', 'neutral'
    """
    text_lower = text.lower()
    words = text_lower.split()

    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0, "neutral"

    score = (pos_count - neg_count) / total

    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return round(score, 2), label


async def analyze_news_sentiment(
    news_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyse le sentiment agrégé d'une liste de news.
    """
    if not news_items:
        return {
            "overall_score": 0.0,
            "overall_label": "neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "items": [],
        }

    scores = []
    analyzed_items = []
    pos_count = neg_count = neutral_count = 0

    for item in news_items:
        text = f"{item['title']} {item.get('body', '')}"
        score, label = analyze_sentiment(text)
        scores.append(score)

        if label == "positive":
            pos_count += 1
        elif label == "negative":
            neg_count += 1
        else:
            neutral_count += 1

        analyzed_items.append({
            "title": item["title"],
            "sentiment_score": score,
            "sentiment_label": label,
            "source": item.get("source", "unknown"),
        })

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.15:
        overall_label = "positive"
    elif avg_score < -0.15:
        overall_label = "negative"
    else:
        overall_label = "neutral"

    return {
        "overall_score": round(avg_score, 2),
        "overall_label": overall_label,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neutral_count,
        "items": analyzed_items[:10],  # Top 10 items
    }
```

**Critères de validation**:
- [ ] Dictionnaires de mots pertinents crypto
- [ ] Score normalisé [-1, 1]
- [ ] Agrégation correcte
- [ ] Tests avec exemples réels

---

## T2.3 - Social Media Monitoring (Optionnel/Future)

### T2.3.1 - Architecture pour Twitter/X (stub)
**Priorité**: BASSE
**Estimation**: À définir

> Note: L'API Twitter est payante. Cette section est un placeholder pour une future intégration.

```python
class TwitterClient:
    """
    Client pour monitorer les mentions Twitter.
    TODO: Implémenter quand API disponible.
    """

    async def get_mentions(self, keywords: List[str]) -> List[Dict]:
        raise NotImplementedError("Twitter API non configurée")

    async def get_influencer_activity(self, handles: List[str]) -> List[Dict]:
        raise NotImplementedError("Twitter API non configurée")
```

**Alternatives gratuites à explorer** :
- Nitter scraping (instable)
- LunarCrush API (tier gratuit limité)
- Santiment (tier gratuit limité)

---

## T2.4 - Snapshot de Tendances Unifié

### T2.4.1 - Implémenter le snapshot de tendances complet
**Priorité**: CRITIQUE
**Estimation**: Moyenne

```python
class TrendAnalyzer:
    """
    Agrégateur de toutes les sources de tendances.
    Fournit un snapshot unifié pour le LLM.
    """

    def __init__(self) -> None:
        self.trends_client = GoogleTrendsClient()
        self.news_client = CryptoNewsClient()

    async def get_trend_snapshot(
        self,
        crypto_keywords: Optional[List[str]] = None,
        socialfi_keywords: Optional[List[str]] = None,
        symbols_to_track: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Génère un snapshot complet des tendances pour le LLM.

        Returns:
        {
            "timestamp": "...",
            "google_trends": {
                "crypto": {...},
                "socialfi": {...},
            },
            "news_sentiment": {
                "overall": {...},
                "by_symbol": {...},
            },
            "trending_narratives": [...],
            "recommendations": [...],
        }
        """
        if crypto_keywords is None:
            crypto_keywords = CRYPTO_KEYWORDS[:5]
        if socialfi_keywords is None:
            socialfi_keywords = SOCIALFI_KEYWORDS[:5]
        if symbols_to_track is None:
            symbols_to_track = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        # Fetch en parallèle
        results = await asyncio.gather(
            self.trends_client.get_interest_over_time(crypto_keywords),
            self.trends_client.get_interest_over_time(socialfi_keywords),
            self.news_client.get_news_cryptocompare(limit=30),
            return_exceptions=True,
        )

        crypto_trends = results[0] if not isinstance(results[0], Exception) else {}
        socialfi_trends = results[1] if not isinstance(results[1], Exception) else {}
        news = results[2] if not isinstance(results[2], Exception) else []

        # Analyser sentiment news
        news_sentiment = await analyze_news_sentiment(news)

        # Sentiment par symbole
        sentiment_by_symbol = {}
        for sym in symbols_to_track:
            sym_news = await self.news_client.search_news_by_symbol(sym, limit=5)
            sentiment_by_symbol[sym] = await analyze_news_sentiment(sym_news)

        # Identifier les narratifs trending
        trending_narratives = self._identify_trending_narratives(
            crypto_trends, socialfi_trends
        )

        # Générer des recommendations basiques
        recommendations = self._generate_recommendations(
            crypto_trends, socialfi_trends, news_sentiment
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "google_trends": {
                "crypto": crypto_trends,
                "socialfi": socialfi_trends,
            },
            "news_sentiment": {
                "overall": news_sentiment,
                "by_symbol": sentiment_by_symbol,
            },
            "trending_narratives": trending_narratives,
            "recommendations": recommendations,
        }

    def _identify_trending_narratives(
        self,
        crypto_trends: Dict,
        socialfi_trends: Dict,
    ) -> List[Dict[str, Any]]:
        """Identifie les narratifs en hausse"""
        narratives = []

        for kw, data in {**crypto_trends, **socialfi_trends}.items():
            if isinstance(data, dict) and data.get("trend") == "rising":
                narratives.append({
                    "keyword": kw,
                    "current_score": data.get("current", 0),
                    "trend": "rising",
                    "related_tokens": KEYWORD_TO_TOKENS.get(kw, []),
                })

        # Trier par score
        narratives.sort(key=lambda x: x["current_score"], reverse=True)
        return narratives[:5]

    def _generate_recommendations(
        self,
        crypto_trends: Dict,
        socialfi_trends: Dict,
        news_sentiment: Dict,
    ) -> List[str]:
        """Génère des recommendations textuelles basiques"""
        recs = []

        # Check sentiment général
        if news_sentiment.get("overall_label") == "positive":
            recs.append("Sentiment news globalement positif - environnement favorable")
        elif news_sentiment.get("overall_label") == "negative":
            recs.append("Sentiment news négatif - prudence recommandée")

        # Check trends rising
        rising_count = sum(
            1 for d in {**crypto_trends, **socialfi_trends}.values()
            if isinstance(d, dict) and d.get("trend") == "rising"
        )
        if rising_count >= 3:
            recs.append(f"{rising_count} narratifs en hausse - opportunités potentielles")

        return recs


# Singleton
_trend_analyzer: Optional[TrendAnalyzer] = None

def get_trend_analyzer() -> TrendAnalyzer:
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    return _trend_analyzer
```

**Critères de validation**:
- [ ] Agrégation de toutes les sources
- [ ] Fetch parallèle pour performance
- [ ] Identification des narratifs trending
- [ ] Recommendations basiques
- [ ] Format optimisé pour le LLM

---

## T2.5 - Analytics Basiques

### T2.5.1 - Implémenter src/tools/analytics.py
**Priorité**: MOYENNE
**Estimation**: Simple

```python
"""
Analytics basiques pour enrichir les données marché.
Calculs de volatilité, regime detection, etc.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_returns(prices: List[float]) -> List[float]:
    """Calcule les returns logarithmiques"""
    if len(prices) < 2:
        return []
    arr = np.array(prices)
    return list(np.diff(np.log(arr)))


def compute_volatility(prices: List[float], annualize: bool = False) -> float:
    """
    Calcule la volatilité (écart-type des returns).
    Si annualize=True, multiplie par sqrt(365*24) pour données horaires.
    """
    returns = compute_returns(prices)
    if not returns:
        return 0.0
    vol = float(np.std(returns))
    if annualize:
        vol *= np.sqrt(365 * 24)  # Pour données horaires
    return round(vol, 4)


def detect_regime(prices: List[float]) -> str:
    """
    Détecte le régime de marché:
    - TREND_UP: tendance haussière
    - TREND_DOWN: tendance baissière
    - RANGE: consolidation
    - CHAOS: haute volatilité sans direction
    """
    if len(prices) < 10:
        return "UNKNOWN"

    returns = compute_returns(prices)
    arr = np.array(returns)

    drift = arr.mean()
    vol = arr.std()

    # Seuils empiriques (à ajuster)
    if vol > 0.05:  # Très volatile
        return "CHAOS"
    if drift > 0.002:  # ~0.2% par période
        return "TREND_UP"
    if drift < -0.002:
        return "TREND_DOWN"
    return "RANGE"


def compute_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calcule le RSI (Relative Strength Index).
    RSI > 70 = surachat, RSI < 30 = survente
    """
    if len(prices) < period + 1:
        return 50.0  # Neutre par défaut

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


def compute_market_analytics(prices: List[float]) -> Dict[str, any]:
    """
    Calcule tous les analytics pour une série de prix.
    """
    return {
        "regime": detect_regime(prices),
        "volatility": compute_volatility(prices),
        "volatility_annualized": compute_volatility(prices, annualize=True),
        "rsi": compute_rsi(prices),
        "price_change_pct": round(
            ((prices[-1] / prices[0]) - 1) * 100, 2
        ) if prices else 0,
        "high": max(prices) if prices else 0,
        "low": min(prices) if prices else 0,
        "data_points": len(prices),
    }
```

**Critères de validation**:
- [ ] Calculs mathématiquement corrects
- [ ] Régimes bien définis
- [ ] RSI standard 14 périodes
- [ ] Volatilité annualisée pour comparaison
- [ ] Tests unitaires avec données connues

---

## Checklist Finale Phase 2

- [ ] GoogleTrendsClient fonctionnel
- [ ] CryptoNewsClient avec sentiment analysis
- [ ] TrendAnalyzer agrégateur
- [ ] Analytics basiques (volatilité, RSI, regime)
- [ ] Keywords crypto/SocialFi configurés
- [ ] Snapshot unifié pour le LLM
- [ ] Tests avec données réelles
- [ ] Gestion des rate limits Google

---

## Notes Techniques

### Rate Limits Google Trends
- Pas de limite officielle mais blocage si abus
- Ajouter des délais entre les requêtes (1-2s)
- Utiliser un VPN si nécessaire en prod

### Amélioration Future du Sentiment
- Intégrer un modèle NLP (FinBERT)
- Ajouter des sources (Reddit, Discord)
- Weighted scoring par source fiabilité

### Performance
- Cache des résultats Google Trends (TTL: 5min)
- Fetch parallèle pour news multi-symboles
- Limiter le nombre de keywords à 5 par requête
