# Policy corpus (Agent 3: Eligibility & Safety)

This folder holds the retailer policy pages and government rules that the policy
RAG answers from. `sources.json` lists every source. Only entries with
`"ingest": true` are fetched and indexed.

## Refresh the corpus

```bash
python scripts/policies/fetch_policies.py        # fetch pages and PDFs into data/policies/<retailer>/
python scripts/policies/build_policy_index.py    # chunk + embed into Chroma (data/chroma/)
```

`fetch_policies.py` never replaces a good file with a broken fetch. If a page
returns too little text (a captcha, a login wall or a JavaScript-only page), it
prints `[too short]` and writes nothing.

## Saving a page by hand

Amazon and Flipkart help pages sometimes block scripted access. When that
happens, open the page in a browser, copy the policy text, and save it as
`data/policies/<retailer>/<id>.md` with this header. Keep the URL and retrieval
date exact, because they appear as citations in answers.

```markdown
---
source_id: amazon-returns-policy
retailer: amazon
doc_type: return_policy
source_url: https://www.amazon.in/gp/help/customer/display.html?nodeId=202111910
retrieved_at: 2026-09-24
---
# Returns Policy

(pasted policy text, keeping its headings)
```

Valid `retailer` values are `amazon`, `flipkart`, `croma`, `reliance_digital`,
`vijay_sales` and `regulation`. The indexer skips files with a missing or bad
header and reports them.

## Keeping copies current

Retailers revise these pages often. The index build, and the app's policy tab,
flag any copy whose `retrieved_at` is older than `POLICY_STALE_DAYS` (default
90) as `[stale  ]`. Re-run `fetch_policies.py`, or save a fresh copy by hand
with today's date.

## Reviewed return windows

`return_windows.json` holds the rules buyers ask about most, copied by hand from
the policy pages here. The Eligibility agent shows the matching row as an exact
answer ("Flipkart, mobile phones: 7 days, replacement only") above the searched
passages. Add a row only from a page you have read, and keep its URL and
retrieval date exact:

```json
{"retailer": "flipkart", "category": "mobile phones", "window_days": 7,
 "action": "replacement", "conditions": "Only if delivered defective or damaged",
 "source_id": "flipkart-return-policy",
 "source_url": "https://www.flipkart.com/pages/returnpolicy",
 "retrieved_at": "2026-09-27"}
```

- `category`: a product type from `CATEGORY_PATTERNS` in
  `tools/eligibility_agent.py` (`mobile phones`, `laptops`, `tablets`, ...), or
  `*` for the retailer's default.
- `action`: `return`, `replacement`, `return_or_replacement`, `service_center`
  or `not_returnable`.

## Checking search quality

After adding sources, add two or three questions for each to
`scripts/policies/eval_policy_questions.py` and run it. A source whose
questions all MISS was probably not fetched, or was fetched as a blocked page.
