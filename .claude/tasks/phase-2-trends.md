# Phase 2 - Trends & Social Sentiment

> **Objectif**: Implémenter l'analyse de tendances (Google Trends) et le sentiment social pour identifier les opportunités SocialFi et memecoins.

## Statut Global
- [ ] Phase complète

## Dépendances
- Phase 0 complète (structure de base)
- Phase 1 partielle (config, base exchange)

**Scope MVP** :
* Google Trends avec mots-clés `socialfi`, `crypto airdrop`, `memecoin`.
* Nouveaux listings MEXC + top gainers 24h MEXC depuis le wrapper exchange.
* Pas de sentiment social avancé (X/Farcaster) ni d'outils V2 pour le moment.

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

---

## T2.6 - Architecture Modulaire Data Providers

> **Objectif**: Assurer que tous les providers de données implémentent les interfaces abstraites.
> Permet de swapper Google Trends ↔ Autre source, CryptoCompare ↔ CoinGecko, etc.

### T2.6.1 - Implémenter GoogleTrendsProvider (BaseTrendsProvider)
**Priorité**: HAUTE
**Estimation**: Simple

Modifier `src/tools/trends.py` pour implémenter l'interface :

```python
"""
Implémentation Google Trends de BaseTrendsProvider.
Modulaire et swappable avec d'autres sources de tendances.
"""

from typing import Any, Dict, List
from src.interfaces import BaseTrendsProvider
from pytrends.request import TrendReq


class GoogleTrendsProvider(BaseTrendsProvider):
    """
    Implémentation concrète de BaseTrendsProvider pour Google Trends.

    Limitations:
    - Max 5 keywords par requête
    - Rate limiting non officiel (risque de blocage)
    - Données relatives, pas absolues
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
        self._client: Optional[TrendReq] = None

    @property
    def name(self) -> str:
        return "google_trends"

    def _get_client(self) -> TrendReq:
        """Lazy initialization du client PyTrends"""
        if self._client is None:
            self._client = TrendReq(
                hl=self._hl,
                tz=self._tz,
                timeout=self._timeout,
                retries=2,
                backoff_factor=0.5,
            )
        return self._client

    async def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "now 7-d",
    ) -> Dict[str, Any]:
        """Implémente BaseTrendsProvider.get_interest_over_time"""
        import asyncio
        loop = asyncio.get_running_loop()

        def _fetch():
            client = self._get_client()
            client.build_payload(keywords[:5], timeframe=timeframe)
            return client.interest_over_time()

        try:
            df = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            return {"error": str(e), "keywords": keywords}

        if df.empty:
            return {kw: {"current": 0, "trend": "unknown"} for kw in keywords}

        result = {}
        for kw in keywords:
            if kw not in df.columns:
                result[kw] = {"current": 0, "trend": "unknown"}
                continue

            series = df[kw].tolist()
            current = series[-1] if series else 0
            avg = sum(series) / len(series) if series else 0

            # Calcul de la tendance
            if len(series) >= 2:
                recent = sum(series[-3:]) / min(3, len(series))
                old = sum(series[:3]) / min(3, len(series))
                if recent > old * 1.1:
                    trend = "rising"
                elif recent < old * 0.9:
                    trend = "falling"
                else:
                    trend = "stable"
            else:
                trend = "unknown"

            result[kw] = {
                "current": int(current),
                "avg": round(avg, 1),
                "trend": trend,
                "series": series[-20:],
            }

        return result

    async def get_related_queries(self, keyword: str) -> Dict[str, List[str]]:
        """Implémente BaseTrendsProvider.get_related_queries"""
        import asyncio
        loop = asyncio.get_running_loop()

        def _fetch():
            client = self._get_client()
            client.build_payload([keyword])
            return client.related_queries()

        try:
            data = await loop.run_in_executor(None, _fetch)
        except Exception as e:
            return {"error": str(e), "rising": [], "top": []}

        result = {"rising": [], "top": []}
        if keyword in data:
            kw_data = data[keyword]
            if "rising" in kw_data and kw_data["rising"] is not None:
                result["rising"] = kw_data["rising"]["query"].tolist()[:10]
            if "top" in kw_data and kw_data["top"] is not None:
                result["top"] = kw_data["top"]["query"].tolist()[:10]

        return result
```

**Critères de validation**:
- [ ] Implémente toutes les méthodes de BaseTrendsProvider
- [ ] Calcul de tendance correct
- [ ] Gestion des erreurs
- [ ] Tests avec données réelles

---

### T2.6.2 - Implémenter CryptoCompareProvider (BaseNewsProvider)
**Priorité**: HAUTE
**Estimation**: Simple

```python
"""
Implémentation CryptoCompare de BaseNewsProvider.
"""

from typing import Any, Dict, List
from datetime import datetime
import requests
import asyncio

from src.interfaces import BaseNewsProvider


class CryptoCompareProvider(BaseNewsProvider):
    """
    Implémentation concrète de BaseNewsProvider pour CryptoCompare.

    Avantages:
    - API gratuite, pas de clé requise
    - Agrège plusieurs sources
    - Mise à jour fréquente
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "OtterTrend/1.0"})
        self._base_url = "https://min-api.cryptocompare.com/data/v2/news/"

    @property
    def name(self) -> str:
        return "cryptocompare"

    async def get_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Implémente BaseNewsProvider.get_news"""
        loop = asyncio.get_running_loop()

        def _fetch():
            resp = self._session.get(
                self._base_url,
                params={"lang": "EN"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("Data", [])[:limit]

        try:
            raw_news = await loop.run_in_executor(None, _fetch)
        except Exception:
            return []

        return [
            {
                "title": item.get("title", ""),
                "body": item.get("body", "")[:500],
                "source": item.get("source_info", {}).get("name", "unknown"),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("published_on", 0)
                ).isoformat(),
                "categories": item.get("categories", "").split("|"),
            }
            for item in raw_news
        ]

    async def search_news_by_symbol(
        self, symbol: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Implémente BaseNewsProvider.search_news_by_symbol"""
        base_asset = symbol.split("/")[0].upper()
        all_news = await self.get_news(limit=100)

        matching = [
            item for item in all_news
            if base_asset in item["title"].upper()
            or base_asset in item["body"].upper()
        ][:limit]

        return matching
```

**Critères de validation**:
- [ ] Implémente toutes les méthodes de BaseNewsProvider
- [ ] Normalisation des données
- [ ] Filtrage par symbole fonctionnel
- [ ] Tests avec mock HTTP

---

### T2.6.3 - Implémenter RuleBasedSentiment (BaseSentimentAnalyzer)
**Priorité**: HAUTE
**Estimation**: Simple

```python
"""
Implémentation basée sur règles de BaseSentimentAnalyzer.
Simple mais efficace pour le domaine crypto.
"""

from typing import Any, Dict, List
from src.interfaces import BaseSentimentAnalyzer


# Dictionnaires de mots-clés crypto
POSITIVE_WORDS = [
    "surge", "rally", "soar", "bullish", "breakout", "ath", "pump",
    "moon", "gains", "growth", "adoption", "partnership", "launch",
    "upgrade", "mainnet", "listing", "approval", "institutional",
]

NEGATIVE_WORDS = [
    "crash", "dump", "bearish", "plunge", "liquidation", "hack",
    "exploit", "scam", "rug", "fraud", "ban", "regulation", "sec",
    "lawsuit", "delay", "failure", "bug", "vulnerability",
]


class RuleBasedSentiment(BaseSentimentAnalyzer):
    """
    Analyseur de sentiment basé sur des règles lexicales.

    Simple mais efficace pour le domaine crypto.
    Peut être remplacé par FinBERT ou GPT pour plus de précision.
    """

    def __init__(
        self,
        positive_words: List[str] = None,
        negative_words: List[str] = None,
    ) -> None:
        self._positive = positive_words or POSITIVE_WORDS
        self._negative = negative_words or NEGATIVE_WORDS

    @property
    def name(self) -> str:
        return "rule_based"

    def analyze(self, text: str) -> Dict[str, Any]:
        """Implémente BaseSentimentAnalyzer.analyze"""
        text_lower = text.lower()

        pos_count = sum(1 for w in self._positive if w in text_lower)
        neg_count = sum(1 for w in self._negative if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        score = (pos_count - neg_count) / total

        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        return {
            "score": round(score, 2),
            "label": label,
            "confidence": min(total / 5, 1.0),  # Plus de mots = plus confiant
            "positive_count": pos_count,
            "negative_count": neg_count,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Implémente BaseSentimentAnalyzer.analyze_batch"""
        return [self.analyze(text) for text in texts]


# Alternative future: FinBERT
class FinBERTSentiment(BaseSentimentAnalyzer):
    """
    TODO: Implémenter avec transformers/FinBERT pour production.
    Plus précis mais plus lent et nécessite GPU.
    """

    def __init__(self) -> None:
        raise NotImplementedError("FinBERT non implémenté - utiliser RuleBasedSentiment")

    def analyze(self, text: str) -> Dict[str, Any]:
        ...

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        ...
```

**Critères de validation**:
- [ ] Implémente toutes les méthodes de BaseSentimentAnalyzer
- [ ] Dictionnaires crypto pertinents
- [ ] Score normalisé [-1, 1]
- [ ] Placeholder pour FinBERT

---

### T2.6.4 - Factory pour data providers
**Priorité**: MOYENNE
**Estimation**: Simple

```python
"""
Factories pour créer les providers de données.
Permet de swapper facilement les implémentations.
"""

from typing import Dict, Type
from src.interfaces import BaseTrendsProvider, BaseNewsProvider, BaseSentimentAnalyzer


# Registres des providers disponibles
TRENDS_REGISTRY: Dict[str, Type[BaseTrendsProvider]] = {}
NEWS_REGISTRY: Dict[str, Type[BaseNewsProvider]] = {}
SENTIMENT_REGISTRY: Dict[str, Type[BaseSentimentAnalyzer]] = {}


def register_trends_provider(name: str):
    def decorator(cls):
        TRENDS_REGISTRY[name] = cls
        return cls
    return decorator


def register_news_provider(name: str):
    def decorator(cls):
        NEWS_REGISTRY[name] = cls
        return cls
    return decorator


def register_sentiment_analyzer(name: str):
    def decorator(cls):
        SENTIMENT_REGISTRY[name] = cls
        return cls
    return decorator


# Enregistrement des implémentations par défaut
@register_trends_provider("google")
class GoogleTrendsProvider(BaseTrendsProvider):
    ...


@register_news_provider("cryptocompare")
class CryptoCompareProvider(BaseNewsProvider):
    ...


@register_sentiment_analyzer("rules")
class RuleBasedSentiment(BaseSentimentAnalyzer):
    ...


# Factories
def create_trends_provider(name: str = "google", **kwargs) -> BaseTrendsProvider:
    if name not in TRENDS_REGISTRY:
        raise ValueError(f"Trends provider '{name}' non supporté")
    return TRENDS_REGISTRY[name](**kwargs)


def create_news_provider(name: str = "cryptocompare", **kwargs) -> BaseNewsProvider:
    if name not in NEWS_REGISTRY:
        raise ValueError(f"News provider '{name}' non supporté")
    return NEWS_REGISTRY[name](**kwargs)


def create_sentiment_analyzer(name: str = "rules", **kwargs) -> BaseSentimentAnalyzer:
    if name not in SENTIMENT_REGISTRY:
        raise ValueError(f"Sentiment analyzer '{name}' non supporté")
    return SENTIMENT_REGISTRY[name](**kwargs)
```

**Critères de validation**:
- [ ] Registres extensibles
- [ ] Factories avec détection automatique
- [ ] Placeholders pour futures implémentations

---

## T2.7 - Accélération Hardware Mac Mini M4

> **Objectif**: Intégrer les optimisations Apple Silicon pour l'analyse de sentiment et les calculs vectoriels.
> Exploite le Neural Engine (38 TOPS), MLX, et Core ML pour des inférences rapides.

### T2.7.1 - Implémenter MLXSentimentAnalyzer (BaseSentimentAnalyzer)
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `src/accelerators/mlx_sentiment.py` avec analyse de sentiment accélérée MLX :

```python
"""
Analyseur de sentiment utilisant MLX pour Apple Silicon.
Charge des modèles optimisés (DistilBERT/FinBERT) via mlx-lm.
"""

from typing import Any, Dict, List, Optional
import sys

from src.interfaces import BaseSentimentAnalyzer
from src.hardware import detect_hardware, ComputeBackend


class MLXSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Analyseur de sentiment utilisant MLX sur Apple Silicon.

    Avantages:
    - Zero-copy unified memory (pas de transfert CPU<->GPU)
    - Lazy evaluation pour optimisation automatique
    - ~5-10x plus rapide que CPU pour transformers

    Modèles supportés:
    - distilbert-base-uncased-finetuned-sst-2-english (général)
    - ProsusAI/finbert (finance/crypto spécialisé)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_length: int = 128,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._initialized = False

        # Vérifier que MLX est disponible
        hw = detect_hardware()
        if not hw.has_mlx:
            raise RuntimeError(
                "MLX non disponible. Utiliser RuleBasedSentiment ou "
                "CoreMLSentimentAnalyzer comme fallback."
            )

    @property
    def name(self) -> str:
        return f"mlx_{self._model_name.split('/')[-1]}"

    def _lazy_init(self) -> None:
        """Initialisation paresseuse du modèle (premier appel)"""
        if self._initialized:
            return

        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load, generate
            from transformers import AutoTokenizer

            # Charger le tokenizer HuggingFace
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Charger le modèle MLX (conversion automatique si nécessaire)
            self._model, _ = load(self._model_name)

            self._initialized = True

        except ImportError as e:
            raise RuntimeError(f"Dépendances MLX manquantes: {e}")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement modèle MLX: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyse le sentiment d'un texte avec MLX.

        Returns:
            {
                "score": float,  # -1.0 à 1.0
                "label": str,    # "positive", "negative", "neutral"
                "confidence": float,
                "probabilities": {"positive": float, "negative": float}
            }
        """
        import mlx.core as mx

        self._lazy_init()

        # Tokeniser
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )

        # Convertir en MLX arrays (zero-copy sur Apple Silicon)
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Inférence
        with mx.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = mx.softmax(logits, axis=-1)

        # Extraire les probabilités (0=négatif, 1=positif pour SST-2)
        probs_np = probs.tolist()[0]
        neg_prob = probs_np[0]
        pos_prob = probs_np[1]

        # Score normalisé [-1, 1]
        score = pos_prob - neg_prob

        # Label
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        # Confidence = écart entre les probas
        confidence = abs(pos_prob - neg_prob)

        return {
            "score": round(score, 3),
            "label": label,
            "confidence": round(confidence, 3),
            "probabilities": {
                "positive": round(pos_prob, 3),
                "negative": round(neg_prob, 3),
            },
            "backend": "mlx",
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyse par batch (optimisé pour MLX lazy evaluation).
        MLX batche automatiquement les opérations pour le GPU.
        """
        import mlx.core as mx

        self._lazy_init()

        if not texts:
            return []

        # Tokeniser tout le batch
        inputs = self._tokenizer(
            texts,
            return_tensors="np",
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )

        # Convertir en MLX
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Inférence batch
        with mx.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask)
            probs = mx.softmax(outputs.logits, axis=-1)

        # Convertir les résultats
        probs_list = probs.tolist()
        results = []

        for probs_item in probs_list:
            neg_prob, pos_prob = probs_item[0], probs_item[1]
            score = pos_prob - neg_prob

            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"

            results.append({
                "score": round(score, 3),
                "label": label,
                "confidence": round(abs(pos_prob - neg_prob), 3),
                "probabilities": {
                    "positive": round(pos_prob, 3),
                    "negative": round(neg_prob, 3),
                },
                "backend": "mlx",
            })

        return results


# Enregistrer dans le registry
from src.tools.factories import register_sentiment_analyzer

@register_sentiment_analyzer("mlx")
class _MLXSentiment(MLXSentimentAnalyzer):
    pass
```

**Critères de validation**:
- [ ] Initialisation paresseuse du modèle
- [ ] Inférence single et batch
- [ ] Score normalisé [-1, 1]
- [ ] Fallback clair si MLX indisponible
- [ ] Tests sur Mac M4

---

### T2.7.2 - Implémenter CoreMLSentimentAnalyzer (BaseSentimentAnalyzer)
**Priorité**: HAUTE
**Estimation**: Moyenne

Créer `src/accelerators/coreml_sentiment.py` avec inférence sur Neural Engine :

```python
"""
Analyseur de sentiment utilisant Core ML sur Neural Engine.
~10x plus rapide que CPU, ~3x plus économe en énergie.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from src.interfaces import BaseSentimentAnalyzer
from src.hardware import detect_hardware


class CoreMLSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Analyseur de sentiment utilisant Core ML (Neural Engine).

    Le Neural Engine du M4 offre 38 TOPS pour l'inférence.
    Idéal pour des analyses fréquentes avec latence minimale.

    Note: Nécessite un modèle pré-converti au format .mlpackage
    Utiliser coremltools pour convertir depuis HuggingFace.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "finbert_sentiment.mlpackage",
        max_length: int = 128,
    ) -> None:
        self._model_path = model_path or str(
            Path(__file__).parent.parent / "models" / model_name
        )
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._initialized = False

        # Vérifier Core ML disponible
        hw = detect_hardware()
        if not hw.has_coreml:
            raise RuntimeError(
                "Core ML non disponible. Utiliser MLXSentimentAnalyzer "
                "ou RuleBasedSentiment comme fallback."
            )

    @property
    def name(self) -> str:
        return "coreml_sentiment"

    def _lazy_init(self) -> None:
        """Charge le modèle Core ML au premier appel"""
        if self._initialized:
            return

        try:
            import coremltools as ct
            from transformers import AutoTokenizer

            # Charger le modèle Core ML
            self._model = ct.models.MLModel(
                self._model_path,
                compute_units=ct.ComputeUnit.ALL  # Inclut Neural Engine
            )

            # Tokenizer correspondant
            self._tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert"  # ou le modèle source
            )

            self._initialized = True

        except FileNotFoundError:
            raise RuntimeError(
                f"Modèle Core ML non trouvé: {self._model_path}. "
                "Exécuter scripts/convert_to_coreml.py d'abord."
            )
        except Exception as e:
            raise RuntimeError(f"Erreur chargement Core ML: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyse avec Neural Engine"""
        import numpy as np

        self._lazy_init()

        # Tokeniser
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )

        # Préparer pour Core ML
        input_dict = {
            "input_ids": inputs["input_ids"].astype(np.int32),
            "attention_mask": inputs["attention_mask"].astype(np.int32),
        }

        # Inférence sur Neural Engine
        outputs = self._model.predict(input_dict)

        # Extraire logits et calculer probas
        logits = outputs["logits"][0]
        probs = self._softmax(logits)

        # FinBERT: [negative, neutral, positive]
        neg_prob, neut_prob, pos_prob = probs

        # Score composite
        score = pos_prob - neg_prob

        # Label
        max_idx = np.argmax(probs)
        labels = ["negative", "neutral", "positive"]
        label = labels[max_idx]

        return {
            "score": round(float(score), 3),
            "label": label,
            "confidence": round(float(probs[max_idx]), 3),
            "probabilities": {
                "positive": round(float(pos_prob), 3),
                "neutral": round(float(neut_prob), 3),
                "negative": round(float(neg_prob), 3),
            },
            "backend": "coreml_neural_engine",
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyse batch séquentielle.
        Note: Core ML supporte le batching mais la tokenization
        doit être gérée individuellement.
        """
        # Pour des performances optimales avec gros batches,
        # considérer MLXSentimentAnalyzer qui gère mieux le batching
        return [self.analyze(text) for text in texts]

    @staticmethod
    def _softmax(x):
        """Softmax numpy"""
        import numpy as np
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# Utilitaire de conversion
def convert_finbert_to_coreml(output_path: str = "models/finbert_sentiment.mlpackage"):
    """
    Convertit FinBERT HuggingFace vers Core ML.
    À exécuter une fois pour préparer le modèle.

    Usage:
        python -c "from src.accelerators.coreml_sentiment import convert_finbert_to_coreml; convert_finbert_to_coreml()"
    """
    import coremltools as ct
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print("Chargement FinBERT depuis HuggingFace...")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    model.eval()

    # Exemple d'entrée pour le trace
    dummy_input = tokenizer(
        "Bitcoin price surges",
        return_tensors="pt",
        max_length=128,
        padding="max_length",
    )

    print("Tracing PyTorch model...")
    traced_model = torch.jit.trace(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
    )

    print("Conversion vers Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 128), dtype=ct.int32),
            ct.TensorType(name="attention_mask", shape=(1, 128), dtype=ct.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,  # Pour M4
    )

    # Métadonnées
    mlmodel.author = "OtterTrend"
    mlmodel.short_description = "FinBERT sentiment analysis for crypto news"
    mlmodel.version = "1.0"

    print(f"Sauvegarde vers {output_path}...")
    mlmodel.save(output_path)
    print("Conversion terminée!")

    return output_path
```

**Critères de validation**:
- [ ] Chargement modèle .mlpackage
- [ ] Inférence sur Neural Engine (ComputeUnit.ALL)
- [ ] Script de conversion HuggingFace → Core ML
- [ ] Fallback si modèle non converti
- [ ] Tests performances vs CPU

---

### T2.7.3 - Accélération des calculs vectoriels avec MLX
**Priorité**: MOYENNE
**Estimation**: Simple

Modifier `src/tools/analytics.py` pour utiliser MLX quand disponible :

```python
"""
Analytics avec accélération MLX optionnelle.
Utilise la mémoire unifiée pour éviter les copies CPU<->GPU.
"""

from typing import Dict, List, Optional
from src.hardware import detect_hardware, ComputeBackend

# Détection au chargement du module
_HW = detect_hardware()
_USE_MLX = _HW.has_mlx and _HW.recommended_backend == ComputeBackend.MLX

if _USE_MLX:
    import mlx.core as mx

    def _to_array(data: List[float]):
        return mx.array(data)

    def _std(arr) -> float:
        return float(mx.std(arr))

    def _mean(arr) -> float:
        return float(mx.mean(arr))

    def _diff(arr):
        return arr[1:] - arr[:-1]

    def _log(arr):
        return mx.log(arr)

    def _where(cond, x, y):
        return mx.where(cond, x, y)

else:
    import numpy as np

    def _to_array(data: List[float]):
        return np.array(data)

    def _std(arr) -> float:
        return float(np.std(arr))

    def _mean(arr) -> float:
        return float(np.mean(arr))

    def _diff(arr):
        return np.diff(arr)

    def _log(arr):
        return np.log(arr)

    def _where(cond, x, y):
        return np.where(cond, x, y)


def compute_returns(prices: List[float]) -> List[float]:
    """Calcule les returns logarithmiques (MLX/NumPy)"""
    if len(prices) < 2:
        return []
    arr = _to_array(prices)
    returns = _diff(_log(arr))
    return list(returns) if not _USE_MLX else returns.tolist()


def compute_volatility(prices: List[float], annualize: bool = False) -> float:
    """Calcule la volatilité avec accélération MLX si disponible"""
    if len(prices) < 2:
        return 0.0

    arr = _to_array(prices)
    log_arr = _log(arr)
    returns = _diff(log_arr)

    vol = _std(returns)

    if annualize:
        # sqrt(365*24) pour données horaires
        vol *= 93.67  # Pre-computed sqrt(8760)

    return round(vol, 4)


def compute_rsi(prices: List[float], period: int = 14) -> float:
    """RSI avec accélération MLX"""
    if len(prices) < period + 1:
        return 50.0

    arr = _to_array(prices)
    deltas = _diff(arr)

    gains = _where(deltas > 0, deltas, 0.0)
    losses = _where(deltas < 0, -deltas, 0.0)

    # Moyenne sur la période
    if _USE_MLX:
        avg_gain = float(mx.mean(gains[-period:]))
        avg_loss = float(mx.mean(losses[-period:]))
    else:
        avg_gain = float(gains[-period:].mean())
        avg_loss = float(losses[-period:].mean())

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


def batch_cosine_similarity_mlx(
    query_embedding: List[float],
    embeddings: List[List[float]],
) -> List[float]:
    """
    Calcul de similarité cosinus batch avec MLX.
    Optimisé pour la recherche dans des embeddings.

    ~10x plus rapide que NumPy sur M4 pour gros batches.
    """
    if not _USE_MLX:
        # Fallback NumPy
        import numpy as np
        q = np.array(query_embedding)
        e = np.array(embeddings)

        q_norm = q / np.linalg.norm(q)
        e_norms = e / np.linalg.norm(e, axis=1, keepdims=True)

        return list(e_norms @ q_norm)

    # MLX optimisé
    q = mx.array(query_embedding)
    e = mx.array(embeddings)

    # Normaliser
    q_norm = q / mx.linalg.norm(q)
    e_norms = e / mx.linalg.norm(e, axis=1, keepdims=True)

    # Dot product batch (exploite le GPU)
    similarities = e_norms @ q_norm

    return similarities.tolist()


# Informations sur le backend utilisé
def get_compute_info() -> Dict[str, any]:
    """Retourne les infos sur le backend de calcul"""
    return {
        "backend": "mlx" if _USE_MLX else "numpy",
        "hardware": _HW.chip_name if _HW.is_apple_silicon else "generic",
        "neural_engine_tops": _HW.neural_engine_tops if _HW.is_apple_silicon else 0,
        "unified_memory": _HW.is_apple_silicon,
    }
```

**Critères de validation**:
- [ ] Fallback transparent NumPy si pas MLX
- [ ] Mêmes résultats MLX vs NumPy (tests)
- [ ] Batch cosine similarity optimisé
- [ ] Fonction get_compute_info() pour debug

---

### T2.7.4 - Adapter TrendAnalyzer pour backends hardware
**Priorité**: MOYENNE
**Estimation**: Simple

Modifier `TrendAnalyzer` pour utiliser le meilleur backend disponible :

```python
# Dans src/tools/trends.py - Modifier TrendAnalyzer

class TrendAnalyzer:
    """
    Agrégateur de tendances avec sélection automatique du backend.
    """

    def __init__(
        self,
        sentiment_backend: str = "auto",  # "auto", "mlx", "coreml", "rules"
    ) -> None:
        self.trends_client = GoogleTrendsProvider()
        self.news_client = CryptoCompareProvider()

        # Sélection du sentiment analyzer
        self._sentiment = self._create_sentiment_analyzer(sentiment_backend)

    def _create_sentiment_analyzer(
        self, backend: str
    ) -> BaseSentimentAnalyzer:
        """
        Crée l'analyseur de sentiment selon le backend demandé.
        "auto" sélectionne le plus performant disponible.
        """
        from src.hardware import detect_hardware, ComputeBackend
        from src.tools.factories import create_sentiment_analyzer

        if backend != "auto":
            return create_sentiment_analyzer(backend)

        # Auto-détection
        hw = detect_hardware()

        # Priorité: CoreML (Neural Engine) > MLX (GPU) > Rules (CPU)
        if hw.has_coreml:
            try:
                return create_sentiment_analyzer("coreml")
            except Exception:
                pass  # Modèle pas converti, fallback

        if hw.has_mlx:
            try:
                return create_sentiment_analyzer("mlx")
            except Exception:
                pass  # Erreur init, fallback

        # Fallback CPU
        return create_sentiment_analyzer("rules")

    async def get_trend_snapshot(self, ...) -> Dict[str, Any]:
        """... (code existant) ..."""
        # Utilise self._sentiment.analyze_batch() pour les news
        ...

    @property
    def sentiment_backend_info(self) -> Dict[str, Any]:
        """Retourne les infos sur le backend sentiment utilisé"""
        from src.tools.analytics import get_compute_info
        return {
            "sentiment_analyzer": self._sentiment.name,
            "compute": get_compute_info(),
        }
```

**Critères de validation**:
- [ ] Auto-détection du meilleur backend
- [ ] Fallback gracieux si erreur
- [ ] Propriété pour debug backend utilisé

---

### T2.7.5 - Script de benchmark backends
**Priorité**: BASSE
**Estimation**: Simple

Créer `scripts/benchmark_sentiment.py` :

```python
#!/usr/bin/env python3
"""
Benchmark des différents backends de sentiment analysis.
Compare CPU (rules), MLX, et Core ML sur M4.
"""

import asyncio
import time
from typing import List, Dict

# Textes de test
TEST_TEXTS = [
    "Bitcoin surges to new all-time high amid institutional adoption",
    "Crypto market crashes as SEC announces new regulations",
    "Ethereum upgrade delayed due to technical issues",
    "Major partnership announced between Solana and major bank",
    "Security vulnerability discovered in popular DeFi protocol",
    # ... ajouter plus de textes
] * 20  # 100 textes pour le benchmark


def benchmark_backend(name: str, analyzer, texts: List[str]) -> Dict:
    """Benchmark un backend"""
    # Warmup
    _ = analyzer.analyze(texts[0])

    # Single inference
    start = time.perf_counter()
    for text in texts[:10]:
        _ = analyzer.analyze(text)
    single_time = (time.perf_counter() - start) / 10

    # Batch inference
    start = time.perf_counter()
    _ = analyzer.analyze_batch(texts)
    batch_time = time.perf_counter() - start

    return {
        "name": name,
        "single_avg_ms": round(single_time * 1000, 2),
        "batch_total_ms": round(batch_time * 1000, 2),
        "batch_per_item_ms": round(batch_time * 1000 / len(texts), 2),
        "texts_per_second": round(len(texts) / batch_time, 1),
    }


def main():
    from src.hardware import detect_hardware
    from src.tools.factories import create_sentiment_analyzer

    hw = detect_hardware()
    print(f"Hardware: {hw.chip_name}")
    print(f"Neural Engine: {hw.neural_engine_tops} TOPS")
    print(f"MLX disponible: {hw.has_mlx}")
    print(f"Core ML disponible: {hw.has_coreml}")
    print("-" * 50)

    results = []

    # Test Rule-based (baseline)
    print("Testing: Rule-based (CPU)...")
    analyzer = create_sentiment_analyzer("rules")
    results.append(benchmark_backend("rules_cpu", analyzer, TEST_TEXTS))

    # Test MLX si disponible
    if hw.has_mlx:
        print("Testing: MLX (GPU)...")
        try:
            analyzer = create_sentiment_analyzer("mlx")
            results.append(benchmark_backend("mlx_gpu", analyzer, TEST_TEXTS))
        except Exception as e:
            print(f"  Erreur MLX: {e}")

    # Test Core ML si disponible
    if hw.has_coreml:
        print("Testing: Core ML (Neural Engine)...")
        try:
            analyzer = create_sentiment_analyzer("coreml")
            results.append(benchmark_backend("coreml_ne", analyzer, TEST_TEXTS))
        except Exception as e:
            print(f"  Erreur Core ML: {e}")

    # Afficher résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS BENCHMARK")
    print("=" * 60)

    for r in results:
        print(f"\n{r['name'].upper()}:")
        print(f"  Single inference: {r['single_avg_ms']} ms/text")
        print(f"  Batch (100 texts): {r['batch_total_ms']} ms total")
        print(f"  Throughput: {r['texts_per_second']} texts/sec")

    # Comparaison
    if len(results) > 1:
        baseline = results[0]["batch_total_ms"]
        print("\n" + "-" * 40)
        print("Speedup vs CPU baseline:")
        for r in results[1:]:
            speedup = baseline / r["batch_total_ms"]
            print(f"  {r['name']}: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
```

**Critères de validation**:
- [ ] Benchmark single et batch
- [ ] Comparaison speedup vs baseline
- [ ] Affichage hardware détecté

---

### T2.7.6 - Mise à jour factory avec backends hardware
**Priorité**: HAUTE
**Estimation**: Simple

Mettre à jour `src/tools/factories.py` :

```python
# Ajouter les imports conditionnels pour les backends hardware

def _register_hardware_backends():
    """Enregistre les backends hardware si disponibles"""
    from src.hardware import detect_hardware

    hw = detect_hardware()

    if hw.has_mlx:
        try:
            from src.accelerators.mlx_sentiment import MLXSentimentAnalyzer
            SENTIMENT_REGISTRY["mlx"] = MLXSentimentAnalyzer
        except ImportError:
            pass

    if hw.has_coreml:
        try:
            from src.accelerators.coreml_sentiment import CoreMLSentimentAnalyzer
            SENTIMENT_REGISTRY["coreml"] = CoreMLSentimentAnalyzer
        except ImportError:
            pass


# Appeler à l'import du module
_register_hardware_backends()
```

**Critères de validation**:
- [ ] Enregistrement conditionnel des backends
- [ ] Pas d'erreur si dépendances manquantes
- [ ] Factory crée le bon backend
