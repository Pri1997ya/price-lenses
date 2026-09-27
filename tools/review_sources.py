"""Collect reviews from Apify review actors, Reddit and YouTube (Agent 3).

Every source is optional and configured from the environment; the fetch script
skips any source that is not configured and says why.

* Amazon / Flipkart: an Apify review actor. Actors differ in their input, so the
  actor id and an input template are both configured; ``{url}`` and ``{limit}``
  in the template are filled in per product::

      APIFY_AMAZON_REVIEWS_ACTOR=<owner>/<actor>
      APIFY_AMAZON_REVIEWS_INPUT={"productUrls": [{"url": "{url}"}], "maxReviews": "{limit}"}

* Reddit: the official API with an app's client id and secret (script app,
  client-credentials grant, read-only).
* YouTube: the Data API v3 with an API key. One search costs 100 quota units
  and each comment page 1 unit.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .review_corpus import (
    Review,
    normalize_marketplace_review,
    normalize_reddit_item,
    normalize_youtube_comment,
)

USER_AGENT = "price-lenses-reviews/1.0 (research; read-only)"
TIMEOUT = 30


class ReviewSourceError(RuntimeError):
    pass


def _fill(template: Any, values: dict[str, Any]) -> Any:
    """Replace "{name}" placeholders anywhere in a JSON-like template."""
    if isinstance(template, dict):
        return {key: _fill(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [_fill(value, values) for value in template]
    if isinstance(template, str):
        for name, value in values.items():
            if template == f"{{{name}}}":
                return value  # keep numbers as numbers
            template = template.replace(f"{{{name}}}", str(value))
    return template


# ----------------------------------------------------------------- Apify ----
@dataclass
class ApifyReviewSource:
    source: str                     # "amazon" or "flipkart"
    actor_id: str
    input_template: dict
    token: str
    client: Any = None

    @classmethod
    def from_env(cls, source: str) -> "ApifyReviewSource | None":
        prefix = f"APIFY_{source.upper()}_REVIEWS"
        actor, template, token = os.getenv(f"{prefix}_ACTOR"), os.getenv(f"{prefix}_INPUT"), os.getenv("APIFY_API_TOKEN")
        if not (actor and template and token):
            return None
        try:
            parsed = json.loads(template)
        except json.JSONDecodeError as exc:
            raise ReviewSourceError(f"{prefix}_INPUT is not valid JSON: {exc}") from None
        return cls(source, actor, parsed, token)

    def fetch(self, product_id: str, url: str, limit: int = 200) -> list[Review]:
        if self.client is None:
            from apify_client import ApifyClient  # lazy import, as in the market provider

            self.client = ApifyClient(self.token)
        from .market_apify_provider import check_run_succeeded, dataset_id_of

        run = self.client.actor(self.actor_id).call(run_input=_fill(self.input_template, {"url": url, "limit": limit}))
        if not run:
            raise ReviewSourceError(f"Apify actor {self.actor_id} returned no run")
        check_run_succeeded(run, self.actor_id)
        dataset_id = dataset_id_of(run)
        if not dataset_id:
            raise ReviewSourceError(f"Apify actor {self.actor_id} run has no default dataset")
        reviews: list[Review] = []
        for item in self.client.dataset(dataset_id).iterate_items():
            # Some actors return one item per review, others one per product with a reviews list.
            entries = item["reviews"] if isinstance(item.get("reviews"), list) else [item]
            for entry in entries:
                review = normalize_marketplace_review(entry, product_id, self.source)
                if review:
                    reviews.append(review)
            if len(reviews) >= limit:
                break
        return reviews[:limit]


# ----------------------------------------------------------------- Reddit ----
@dataclass
class RedditSource:
    client_id: str
    client_secret: str
    subreddits: str = ""            # "IndianGaming+GadgetsIndia"; empty searches all of Reddit
    session: Any = None
    _token: str | None = None

    @classmethod
    def from_env(cls) -> "RedditSource | None":
        client_id, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
        if not (client_id and secret):
            return None
        return cls(client_id, secret, os.getenv("REDDIT_SUBREDDITS", ""))

    def _get(self, path: str, params: dict) -> Any:
        self.session = self.session or requests.Session()
        if not self._token:
            response = self.session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
            )
            response.raise_for_status()
            self._token = response.json()["access_token"]
        response = self.session.get(
            f"https://oauth.reddit.com{path}", params=params,
            headers={"Authorization": f"bearer {self._token}", "User-Agent": USER_AGENT}, timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def fetch(self, product_id: str, query: str, limit: int = 50, comment_posts: int = 5) -> list[Review]:
        path = f"/r/{self.subreddits}/search" if self.subreddits else "/search"
        params = {"q": f'"{query}"', "sort": "relevance", "t": "year", "limit": min(limit, 100), "type": "link"}
        if self.subreddits:
            params["restrict_sr"] = "1"
        posts = self._get(path, params).get("data", {}).get("children", [])
        reviews = [r for r in (normalize_reddit_item(post, product_id) for post in posts) if r]
        # Owners describe problems in the comments more than in the posts.
        for post in posts[:comment_posts]:
            post_id = post.get("data", {}).get("id")
            if not post_id:
                continue
            listing = self._get(f"/comments/{post_id}", {"limit": 100, "depth": 1, "sort": "top"})
            comments = listing[1]["data"]["children"] if isinstance(listing, list) and len(listing) > 1 else []
            reviews.extend(
                r for r in (normalize_reddit_item(c, product_id) for c in comments if c.get("kind") == "t1") if r
            )
        return reviews


# ---------------------------------------------------------------- YouTube ----
@dataclass
class YouTubeSource:
    api_key: str
    session: Any = None

    @classmethod
    def from_env(cls) -> "YouTubeSource | None":
        key = os.getenv("YOUTUBE_API_KEY")
        return cls(key) if key else None

    def _get(self, endpoint: str, params: dict) -> dict:
        self.session = self.session or requests.Session()
        response = self.session.get(
            f"https://www.googleapis.com/youtube/v3/{endpoint}",
            params={**params, "key": self.api_key}, timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def fetch(self, product_id: str, query: str, videos: int = 3, per_video: int = 100) -> list[Review]:
        found = self._get("search", {
            "part": "snippet", "q": f"{query} long term review", "type": "video",
            "maxResults": videos, "regionCode": "IN", "relevanceLanguage": "en",
        })
        reviews: list[Review] = []
        for item in found.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            try:
                threads = self._get("commentThreads", {
                    "part": "snippet", "videoId": video_id, "maxResults": min(per_video, 100),
                    "order": "relevance", "textFormat": "plainText",
                })
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    continue  # comments disabled on this video
                raise
            for thread in threads.get("items", []):
                thread.setdefault("snippet", {}).setdefault("videoId", video_id)
                review = normalize_youtube_comment(thread, product_id)
                if review:
                    reviews.append(review)
        return reviews


def configured_sources() -> dict[str, Callable | None]:
    """Source name -> configured collector, or None when not configured."""
    return {
        "amazon": ApifyReviewSource.from_env("amazon"),
        "flipkart": ApifyReviewSource.from_env("flipkart"),
        "reddit": RedditSource.from_env(),
        "youtube": YouTubeSource.from_env(),
    }
