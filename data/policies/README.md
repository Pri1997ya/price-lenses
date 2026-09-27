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
