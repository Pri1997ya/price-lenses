"""Agent 2 implementation for current Indian market intelligence."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from .market_agent_models import FreshnessPolicy, MarketAgentRequest, SUPPORTED_RETAILERS
from .market_agent_tools import MarketAgentTools, variant_facets, variant_match_tier
from .market_analysis import age_minutes, build_market_report, freshness_for_rows, marketplace_name
from .market_verifier import verify_market_report


SummaryWriter = Callable[[dict[str, Any]], str]


class MarketInvestigatorAgent:
    """Live-aware Agent 2 with database fallback and grounded output."""

    def __init__(
        self,
        database: Any,
        providers: list[Any] | None = None,
        *,
        freshness: FreshnessPolicy | None = None,
        summary_writer: SummaryWriter | None = None,
        enable_llm_summary: bool = False,
    ):
        self.database = database
        self.providers = list(providers or [])
        self.freshness = freshness or FreshnessPolicy()
        self.tools = MarketAgentTools(database, self.providers)
        self.summary_writer = summary_writer
        self.enable_llm_summary = enable_llm_summary

    def analyze(self, request: MarketAgentRequest, *, limit: int = 20) -> dict[str, Any]:
        started_at = perf_counter()
        requested_attributes = {
            key: value
            for key, value in variant_facets(request.query).items()
            if value is not None
        }
        trace: list[dict[str, Any]] = [self._trace_event(
            stage="understand_request",
            tool="parse_product_request",
            source="deterministic",
            input_summary="User product query",
            output_summary=(
                "Product family identified; requested attributes: "
                + (", ".join(f"{key}={value}" for key, value in requested_attributes.items())
                   or "none — expand all verified variants")
            ),
            duration_ms=(perf_counter() - started_at) * 1000,
        )]
        provider_runs: list[dict[str, Any]] = []
        warnings: list[str] = []
        selected: list[Any] = []
        skipped: list[str] = []
        refresh_reason: str | None = None
        source_mode = "database_only"
        fallback_retailers: list[str] = []
        live_offer_count = 0

        if request.provider_policy == "api_first":
            selected = list(self.providers)
            refresh_reason = "api-first live lookup"
            if selected:
                provider_started = perf_counter()
                provider_runs = self.tools.refresh_market_snapshot(
                    request.query, selected, limit=limit
                )
                trace.append(self._trace_event(
                    stage="search_current_market",
                    tool="search_serpapi_and_apify",
                    source="provider",
                    input_summary=f"Query plus result limit {limit}",
                    output_summary=self._provider_result_summary(provider_runs),
                    duration_ms=(perf_counter() - provider_started) * 1000,
                ))
                warnings.extend(self._provider_warnings(provider_runs))
                run_ids = [
                    run["run_id"]
                    for run in provider_runs
                    if run.get("run_id") and run.get("status") != "error" and run.get("count", 0) > 0
                ]
                live_snapshot = self.tools.get_market_snapshot(
                    request.query, request.canonical_id, run_ids=run_ids
                ) if run_ids else {"product": {}, "offers": [], "promotions": []}
                live_offer_count = len(live_snapshot["offers"])
            else:
                live_snapshot = {"product": {}, "offers": [], "promotions": []}
                warnings.append(
                    "No market provider is configured; using stored PostgreSQL evidence."
                )
                trace.append(self._trace_event(
                    stage="search_current_market",
                    tool="search_serpapi_and_apify",
                    source="provider",
                    input_summary="No configured market provider",
                    output_summary="Provider search skipped; PostgreSQL fallback used",
                    duration_ms=0,
                    status="skipped",
                ))

            database_started = perf_counter()
            stored_snapshot = self.tools.get_market_snapshot(request.query, None)
            trace.append(self._trace_event(
                stage="load_market_data",
                tool="get_market_snapshot",
                source="database",
                input_summary="Latest stored observations for the product family",
                output_summary=f"Loaded {len(stored_snapshot.get('offers') or [])} stored offers",
                duration_ms=(perf_counter() - database_started) * 1000,
            ))
            if live_snapshot["offers"]:
                snapshot = self._merge_snapshots(live_snapshot, stored_snapshot)
                live_markets = {
                    marketplace_name(row.get("marketplace"))
                    for row in live_snapshot["offers"]
                }
                stored_markets = {
                    marketplace_name(row.get("marketplace"))
                    for row in stored_snapshot["offers"]
                }
                fallback_retailers = sorted(
                    market for market in stored_markets - live_markets if market
                )
                source_mode = (
                    "live_with_database_fallback" if fallback_retailers else "live_primary"
                )
                if fallback_retailers:
                    warnings.append(
                        "Live APIs did not return every retailer; stored observations filled: "
                        + ", ".join(fallback_retailers)
                    )
            else:
                snapshot = stored_snapshot
                source_mode = "database_fallback"
                if snapshot["offers"]:
                    warnings.append(
                        "Live APIs returned no comparable offers; using stored observations."
                    )
        elif request.provider_policy == "database_only":
            database_started = perf_counter()
            snapshot = self.tools.get_market_snapshot(request.query, request.canonical_id)
            trace.append(self._trace_event(
                stage="load_market_data",
                tool="get_market_snapshot",
                source="database",
                input_summary="Product query and optional canonical ID",
                output_summary=f"Loaded {len(snapshot.get('offers') or [])} stored offers",
                duration_ms=(perf_counter() - database_started) * 1000,
            ))
        else:
            database_started = perf_counter()
            snapshot = self.tools.get_market_snapshot(request.query, request.canonical_id)
            trace.append(self._trace_event(
                stage="load_market_data",
                tool="get_market_snapshot",
                source="database",
                input_summary="Product query and optional canonical ID",
                output_summary=f"Loaded {len(snapshot.get('offers') or [])} stored offers",
                duration_ms=(perf_counter() - database_started) * 1000,
            ))
            refresh_reason = self._refresh_reason(snapshot)
            selected, skipped = self._providers_to_refresh(
                request.query,
                reason=refresh_reason,
                force=request.force_refresh,
            )
            source_mode = "database_primary"

        if request.provider_policy == "database_first" and selected:
            provider_started = perf_counter()
            provider_runs = self.tools.refresh_market_snapshot(
                request.query, selected, limit=limit
            )
            trace.append(self._trace_event(
                stage="search_current_market",
                tool="search_serpapi_and_apify",
                source="provider",
                input_summary=f"Query plus result limit {limit}",
                output_summary=self._provider_result_summary(provider_runs),
                duration_ms=(perf_counter() - provider_started) * 1000,
            ))
            warnings.extend(self._provider_warnings(provider_runs))
            snapshot = self.tools.get_market_snapshot(request.query, request.canonical_id)
            source_mode = "database_first_refreshed"
        elif request.provider_policy == "database_first" and refresh_reason and not self.providers:
            warnings.append(
                "Stored market evidence is incomplete or stale and no configured provider "
                "was available for refresh."
            )
        if skipped:
            warnings.append(
                "Skipped recently attempted provider refresh: " + ", ".join(sorted(skipped))
            )

        grouping_started = perf_counter()
        product = snapshot["product"] or {
            "canonical_id": request.canonical_id,
            "title": request.query,
        }
        sales = self.tools.get_upcoming_sales(request.deadline_days)
        context = {
            "bank": request.bank,
            "card_type": request.card_type,
            "wants_emi": request.wants_emi,
        }
        variant_groups = self._build_variant_groups(
            request.query,
            snapshot.get("variant_options") or [],
            snapshot.get("offers") or [],
            snapshot.get("promotions") or [],
            sales,
            provider_runs,
            context,
        )
        report = self._build_family_report(
            product=product,
            groups=variant_groups,
            provider_runs=provider_runs,
            sales=sales,
            warnings=warnings,
            requested_attributes=requested_attributes,
        )
        trace.append(self._trace_event(
            stage="validate_and_group_variants",
            tool="group_product_variants",
            source="deterministic",
            input_summary=f"{len(snapshot.get('offers') or [])} candidate offers",
            output_summary=f"Created {len(variant_groups)} verified exact-variant groups",
            duration_ms=(perf_counter() - grouping_started) * 1000,
        ))
        report["variant_options"] = [
            {
                key: option.get(key)
                for key in (
                    "canonical_id", "title", "brand", "model", "lowest_price",
                    "observed_offer_count", "variant", "relevance",
                )
            }
            for option in snapshot.get("variant_options", [])
        ]
        report["unresolved_variant_fields"] = []
        report["refresh"] = {
            "policy": request.provider_policy,
            "requested": request.force_refresh or request.provider_policy == "api_first",
            "performed": bool(selected),
            "reason": "forced" if request.force_refresh else refresh_reason,
            "providers": [provider.name for provider in selected],
            "source_mode": source_mode,
            "live_offer_count": live_offer_count,
            "database_fallback_used": source_mode in {
                "database_fallback", "live_with_database_fallback"
            },
            "database_fallback_retailers": fallback_retailers,
        }

        verification_started = perf_counter()
        verify_market_report(report, snapshot["offers"], snapshot["promotions"])
        trace.append(self._trace_event(
            stage="verify_market_report",
            tool="verify_market_report",
            source="deterministic",
            input_summary=f"{len(variant_groups)} variant groups and their offer-bound promotions",
            output_summary="All displayed prices and promotions are grounded",
            duration_ms=(perf_counter() - verification_started) * 1000,
        ))
        report["agent_trace"] = trace
        self._add_llm_summary(report)
        return report

    @staticmethod
    def _trace_event(
        *,
        stage: str,
        tool: str,
        source: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        status: str = "completed",
        display_prompt: str | None = None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "tool": tool,
            "status": status,
            "source": source,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": round(max(0.0, duration_ms), 1),
            "display_prompt": display_prompt,
        }

    @staticmethod
    def _provider_result_summary(runs: list[dict[str, Any]]) -> str:
        count = sum(int(run.get("count") or 0) for run in runs)
        succeeded = sum(run.get("status") != "error" for run in runs)
        return f"{count} offers stored from {succeeded}/{len(runs)} provider runs"

    @staticmethod
    def _merge_snapshots(
        primary: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge current live observations with stored coverage by stable identifiers."""
        offers: dict[str, dict[str, Any]] = {}
        for row in (primary.get("offers") or []) + (fallback.get("offers") or []):
            key = str(row.get("offer_id") or (
                row.get("canonical_id"), row.get("marketplace"), row.get("external_id")
            ))
            offers.setdefault(key, row)
        promotions: dict[str, dict[str, Any]] = {}
        for row in (primary.get("promotions") or []) + (fallback.get("promotions") or []):
            key = str(row.get("promotion_id") or (
                row.get("offer_id"), row.get("description")
            ))
            promotions.setdefault(key, row)
        options: dict[str, dict[str, Any]] = {}
        for row in (primary.get("variant_options") or []) + (fallback.get("variant_options") or []):
            options.setdefault(str(row.get("canonical_id")), row)
        return {
            "product": primary.get("product") or fallback.get("product") or {},
            "offers": list(offers.values()),
            "promotions": list(promotions.values()),
            "variant_options": list(options.values()),
            "unresolved_variant_fields": [],
        }

    def _build_variant_groups(
        self,
        query: str,
        options: list[dict[str, Any]],
        offers: list[dict[str, Any]],
        promotions: list[dict[str, Any]],
        sales: list[dict[str, Any]],
        provider_runs: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requested = variant_facets(query)
        option_by_id = {str(item.get("canonical_id")): item for item in options}
        for row in offers:
            identifier = str(row.get("canonical_id"))
            option_by_id.setdefault(identifier, {
                "canonical_id": identifier,
                "title": row.get("product_title") or row.get("title"),
                "variant": variant_facets(row.get("product_title") or row.get("title")),
            })

        buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
        for identifier, option in option_by_id.items():
            facets = option.get("variant") or variant_facets(option.get("title"))
            signature_values = tuple(facets.get(field) for field in (
                "storage", "ram", "color", "connectivity", "screen_size",
                "device_size", "generation", "condition", "bundle",
            ))
            signature = signature_values if any(signature_values) else (identifier,)
            bucket = buckets.setdefault(signature, {
                "options": [], "canonical_ids": set(), "variant": facets,
            })
            bucket["options"].append(option)
            bucket["canonical_ids"].add(identifier)

        groups: list[dict[str, Any]] = []
        for signature, bucket in buckets.items():
            member_offers = [
                row for row in offers
                if str(row.get("canonical_id")) in bucket["canonical_ids"]
            ]
            if not member_offers:
                continue
            offer_ids = {str(row.get("offer_id")) for row in member_offers}
            member_promotions = [
                row for row in promotions if str(row.get("offer_id")) in offer_ids
            ]
            reference = max(
                bucket["options"], key=lambda item: float(item.get("relevance") or 0)
            )
            variant_report = build_market_report(
                product=reference,
                offers=member_offers,
                promotions=member_promotions,
                sales=sales,
                provider_runs=provider_runs,
                policy=self.freshness,
                context=context,
            ).to_dict()
            if not variant_report.get("ranked_offers"):
                continue
            group = {
                "variant_signature": " | ".join(
                    str(value) for value in signature if value
                ),
                "match_type": variant_match_tier(requested, bucket["variant"]),
                "title": reference.get("title"),
                "canonical_ids": sorted(bucket["canonical_ids"]),
                "variant": bucket["variant"],
                "best_unconditional_offer": variant_report.get("best_unconditional_offer"),
                "best_conditional_offer": variant_report.get("best_conditional_offer"),
                "ranked_offers": variant_report.get("ranked_offers") or [],
                "coverage": variant_report.get("coverage") or {},
                "freshness": variant_report.get("freshness") or {},
                "offer_count": len(variant_report.get("ranked_offers") or []),
                "promotion_count": sum(
                    int(item.get("promotion_count") or 0)
                    for item in variant_report.get("ranked_offers") or []
                ),
                "missing_inputs": variant_report.get("missing_inputs") or [],
                "evidence": variant_report.get("evidence") or [],
            }
            groups.append(group)
        tier_order = {
            "exact_variant": 0,
            "same_configuration_other_color": 1,
            "other_configuration": 2,
        }
        groups.sort(key=lambda group: (
            tier_order.get(group["match_type"], 9),
            float((group.get("best_unconditional_offer") or {}).get("price") or float("inf")),
        ))
        return groups

    @staticmethod
    def _build_family_report(
        *,
        product: dict[str, Any],
        groups: list[dict[str, Any]],
        provider_runs: list[dict[str, Any]],
        sales: list[dict[str, Any]],
        warnings: list[str],
        requested_attributes: dict[str, Any],
    ) -> dict[str, Any]:
        if not groups:
            report = build_market_report(
                product=product, offers=[], promotions=[], sales=sales,
                provider_runs=provider_runs, warnings=warnings,
            ).to_dict()
            report.update({
                "match_mode": "product_family_not_found",
                "requested_attributes": requested_attributes,
                "requested_match": None,
                "variant_groups": [],
                "lowest_starting_price": None,
            })
            return report

        requested_match = next(
            (group for group in groups if group["match_type"] == "exact_variant"), None
        )
        primary = requested_match or groups[0]
        all_ranked = [offer for group in groups for offer in group["ranked_offers"]]
        all_evidence = [item for group in groups for item in group["evidence"]]
        unconditional = [
            group["best_unconditional_offer"] for group in groups
            if group.get("best_unconditional_offer")
        ]
        conditional = [
            group["best_conditional_offer"] for group in groups
            if group.get("best_conditional_offer")
        ]
        verified = sorted({
            retailer
            for group in groups
            for retailer in group.get("coverage", {}).get("verified_retailers", [])
        })
        report = {
            "schema_version": "1.1",
            "agent": "market_investigator",
            "status": "complete" if len(verified) == len(SUPPORTED_RETAILERS) else "partial",
            "product": product,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "provider_runs": provider_runs,
            "coverage": {
                "expected_retailers": list(SUPPORTED_RETAILERS),
                "verified_retailers": verified,
                "missing_retailers": [item for item in SUPPORTED_RETAILERS if item not in verified],
            },
            "freshness": primary.get("freshness") or {},
            "best_listed_offer": min(unconditional, key=lambda item: item["price"], default=None),
            "best_unconditional_offer": min(unconditional, key=lambda item: item["price"], default=None),
            "best_conditional_offer": min(conditional, key=lambda item: item["price"], default=None),
            "ranked_offers": all_ranked,
            "upcoming_sales": sales,
            "signals": ["BEST_CURRENT_VERIFIED_PRICE"],
            "confidence": round(sum(
                len(group.get("coverage", {}).get("verified_retailers", []))
                for group in groups
            ) / max(1, len(groups) * len(SUPPORTED_RETAILERS)), 2),
            "missing_inputs": sorted({
                item
                for group in groups
                for item in group.get("missing_inputs") or []
            }),
            "warnings": list(dict.fromkeys(warnings)),
            "evidence": all_evidence,
            "match_mode": (
                "exact_variant" if requested_match
                else "expanded_colors" if requested_attributes and any(
                    group["match_type"] == "same_configuration_other_color" for group in groups
                )
                else "expanded_variants"
            ),
            "requested_attributes": requested_attributes,
            "requested_match": requested_match,
            "variant_groups": groups,
            "lowest_starting_price": min(item["price"] for item in unconditional),
            "summary": (
                f"{product.get('title') or 'This product'} starts at "
                f"₹{min(item['price'] for item in unconditional):,.0f} across "
                f"{len(groups)} verified variant{'s' if len(groups) != 1 else ''}."
            ),
        }
        if requested_attributes and not requested_match:
            report["warnings"].append(
                "The exact requested configuration was not verified; closest same-product variants are shown."
            )
        elif requested_match:
            exact = requested_match.get("best_unconditional_offer") or {}
            report["summary"] = (
                f"The requested variant is available from ₹{float(exact['price']):,.0f} "
                f"at {exact.get('retailer')}. Other verified variants start at "
                f"₹{report['lowest_starting_price']:,.0f}."
            )
        return report

    @staticmethod
    def _provider_warnings(provider_runs: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        for run in provider_runs:
            if run.get("error"):
                warnings.append(f"{run['provider']}: {run['error']}")
            warnings.extend(
                f"{run['provider']}: {warning}" for warning in run.get("warnings", [])
            )
        return warnings

    def _refresh_reason(self, snapshot: dict[str, Any]) -> str | None:
        rows = snapshot.get("offers") or []
        if not rows:
            return "no stored offers"
        freshness = freshness_for_rows(rows, self.freshness)
        if freshness["status"] in {"stale", "mixed", "empty"}:
            return f"{freshness['status']} stored prices"
        covered = {
            marketplace_name(row.get("marketplace"))
            for row in rows
            if row.get("price") is not None
        }
        if set(SUPPORTED_RETAILERS) - covered:
            return "missing retailer coverage"
        available = [row for row in rows if row.get("availability")]
        if not available:
            return "missing availability evidence"
        if any(
            (age_minutes(row.get("fetched_at")) or 0) > self.freshness.availability_minutes
            for row in available
        ):
            return "stale availability evidence"
        delivered = [row for row in rows if row.get("delivery_by") or row.get("shipping")]
        if delivered and any(
            (age_minutes(row.get("fetched_at")) or 0) > self.freshness.delivery_minutes
            for row in delivered
        ):
            return "stale delivery evidence"
        promotions = snapshot.get("promotions") or []
        if not promotions:
            return "missing promotion evidence"
        if any(
            (age_minutes(row.get("fetched_at")) or 0) > self.freshness.promotion_minutes
            for row in promotions
        ):
            return "stale promotion evidence"
        return None

    def _providers_to_refresh(
        self,
        query: str,
        *,
        reason: str | None,
        force: bool,
    ) -> tuple[list[Any], list[str]]:
        if not force and not reason:
            return [], []
        recent_runs = (
            self.database.recent_runs_for_query(query)
            if hasattr(self.database, "recent_runs_for_query")
            else []
        )
        latest_by_provider: dict[str, dict[str, Any]] = {}
        for run in recent_runs:
            latest_by_provider.setdefault(str(run.get("provider")), run)
        now = datetime.now(timezone.utc)
        selected, skipped = [], []
        candidates = self.providers
        if not force and reason and any(
            evidence in reason
            for evidence in ("promotion", "availability", "delivery", "seller")
        ):
            apify = [provider for provider in self.providers if provider.name == "apify"]
            if apify:
                candidates = apify
        cooldown = self._refresh_cooldown(reason)
        for provider in candidates:
            if force:
                selected.append(provider)
                continue
            recent = latest_by_provider.get(provider.name)
            started = recent.get("started_at") if recent else None
            if isinstance(started, str):
                try:
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                except ValueError:
                    started = None
            if isinstance(started, datetime):
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age = (now - started.astimezone(timezone.utc)).total_seconds() / 60
                if age <= cooldown:
                    skipped.append(provider.name)
                    continue
            selected.append(provider)
        return selected, skipped

    def _refresh_cooldown(self, reason: str | None) -> int:
        if not reason:
            return self.freshness.price_minutes
        if "promotion" in reason:
            return self.freshness.promotion_minutes
        if "availability" in reason:
            return self.freshness.availability_minutes
        if "delivery" in reason:
            return self.freshness.delivery_minutes
        if "seller" in reason:
            return self.freshness.seller_minutes
        return self.freshness.price_minutes

    def _add_llm_summary(self, report: dict[str, Any]) -> None:
        writer = self.summary_writer
        if writer is None and self.enable_llm_summary:
            writer = self._default_llm_summary
        if writer is None:
            report.setdefault("agent_trace", []).append(self._trace_event(
                stage="prepare_user_summary",
                tool="generate_market_summary",
                source="llm",
                input_summary="Verified Agent 2 variant comparison",
                output_summary="LLM summary disabled; deterministic summary displayed",
                duration_ms=0,
                status="skipped",
                display_prompt=self._summary_display_prompt(),
            ))
            return
        started_at = perf_counter()
        try:
            summary = " ".join(str(writer(report)).split())
            words = summary.split()
            if len(words) > 150:
                summary = " ".join(words[:150])
            upper = summary.upper().replace("-", " ").replace("_", " ")
            if any(decision in upper for decision in ("BUY NOW", "BUY ABROAD", "WAIT")):
                raise ValueError("Agent 2 summary attempted to make a final decision")
            if summary:
                report["summary"] = summary
            report.setdefault("agent_trace", []).append(self._trace_event(
                stage="prepare_user_summary",
                tool="generate_market_summary",
                source="llm",
                input_summary=(
                    f"{len(report.get('variant_groups') or [])} verified variant groups"
                ),
                output_summary="Grounded market summary generated",
                duration_ms=(perf_counter() - started_at) * 1000,
                display_prompt=self._summary_display_prompt(),
            ))
        except Exception as exc:  # deterministic report remains usable without the LLM
            report["warnings"].append(f"LLM summary unavailable: {exc}")
            report.setdefault("agent_trace", []).append(self._trace_event(
                stage="prepare_user_summary",
                tool="generate_market_summary",
                source="llm",
                input_summary=(
                    f"{len(report.get('variant_groups') or [])} verified variant groups"
                ),
                output_summary=(
                    "LLM unavailable; deterministic summary retained "
                    f"({type(exc).__name__})"
                ),
                duration_ms=(perf_counter() - started_at) * 1000,
                status="fallback",
                display_prompt=self._summary_display_prompt(),
            ))

    @staticmethod
    def _summary_display_prompt() -> str:
        return (
            "Explain the verified Indian-market price comparison. Prioritize an explicitly "
            "requested variant, then summarize other verified variants. Use only supplied "
            "prices, sellers, promotions, freshness, and coverage. Do not invent facts, "
            "assess seller safety, or make BUY NOW / WAIT decisions."
        )

    @staticmethod
    def _default_llm_summary(report: dict[str, Any]) -> str:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:5001/gateway/mlflow/v1"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            model=os.getenv("LLM_MODEL", "gemini"),
            timeout=20,
            max_retries=1,
        )
        facts = {
            key: report.get(key)
            for key in (
                "product", "coverage", "freshness", "upcoming_sales", "signals",
                "confidence", "missing_inputs", "match_mode", "requested_attributes",
                "lowest_starting_price", "warnings",
            )
        }
        facts["variant_groups"] = [
            {
                "title": group.get("title"),
                "match_type": group.get("match_type"),
                "variant": group.get("variant"),
                "best_unconditional_offer": group.get("best_unconditional_offer"),
                "best_conditional_offer": group.get("best_conditional_offer"),
                "offer_count": group.get("offer_count"),
                "promotion_count": group.get("promotion_count"),
                "verified_retailers": (group.get("coverage") or {}).get(
                    "verified_retailers", []
                ),
            }
            for group in report.get("variant_groups") or []
        ]
        prompt = (
            "You are PriceLens Agent 2 for India. "
            + MarketInvestigatorAgent._summary_display_prompt()
            + " Respond in at most 150 words.\n\n" +
            json.dumps(facts, default=str)
        )
        response = llm.invoke(prompt)
        return str(response.content)
