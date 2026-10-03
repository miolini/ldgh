# Matter Review MVP

A read-only dashboard prototype for reviewing the Sapini matter: overdue work, upcoming dates, case issues with source links, and an interactive 3D activity map.

## Run locally

From this folder, start a static server:

```sh
python3 server.py
```

Then open <http://localhost:8000>. The page loads `sapini-clio-data.json` from this folder and imports Three.js from jsDelivr, so an internet connection is required for the 3D scene and web fonts.

The dashboard server proxies chat requests to an OpenAI-compatible local inference service. The provider base URL defaults to `http://100.99.215.85:8082/v1` and can be changed with `FIELDNOTE_LLM_BASE_URL`; the expected model ID is `qwen3.8-27b-uncensored`. The browser only calls the same-origin `/api/llm/models` and `/api/llm/chat/completions` routes, and streamed responses pass through as they arrive. Case context is sent from the dashboard server to the configured provider over the local network.

The earlier weight cache remains resumable at `/api/model-cache/status` and `/api/model-cache/start`, but the dashboard no longer starts a weight download. The existing partial cache is kept on disk.

## Prototype scope

- Review open tasks in due-date order, with overdue labels.
- See upcoming calendar events and source-backed case issues.
- Scrub the matter timeline, filter notes, communications, tasks, calendar events, documents, and expenses, then inspect a selected record.
- Open a provider-view preview; it is a local mock and does not share or transmit data.
- Run 1,000,000 Monte Carlo trials in a browser background worker, with live progress, outcome distributions, and investigation priorities ranked by model sensitivity.
- Adjust evidence weights and scenario assumptions, including dependence between liability and employment scope. Same-source evidence receives a partial correlation discount.
- Uses the local JSON seed snapshot, not a live Clio connection. It does not write to Clio.

## Case chat

The dashboard chat automatically prepares the case prefix from the snapshot and sends it through the backend proxy. Responses stream into the chat panel and support Markdown. The local agent can make up to 100 tool actions per reply to search records, inspect model inputs, add or update investigator facts and assumptions, and run simulations. Conversation context retains up to eight recent turns and preserves complete tool exchanges. Confirm important dates, amounts, and procedural statements against the source records.

The outcome model uses illustrative analyst inputs and has not been calibrated to actual case outcomes. Investigation priorities are sensitivity estimates rather than fraud findings or expected case values.


## Expanded case network

The Outcomes page exposes 34 nodes and 43 dependency links. Eleven evidence-adjusted parameters cover collision mechanism, employment scope, injury causation, treatment continuity, ledger support, income documentation, work-loss attribution, surgery necessity, surgery completion, lasting impairment and collection success. A shared causation event links the medical, income and care branches. Supported losses flow through comparative fault, a resolution factor, coverage, collection, lien deduction and an optional time discount.

Assumptions are grouped by branch. All event priors, triangular cost/fraction ranges and Beta concentrations are editable; contradictory low/mode/high ranges are rejected. Existing saved inputs are retained, with defaults for newly introduced parameters. New parameters have illustrative priors rather than weights inferred from case records. Evidence updates can target each parameter independently. The local agent uses the same parameter definitions and can inspect inputs, node estimates, controls or investigation priorities through separate tool responses.

One million trials run in a background worker. Each 100,000-trial update aggregates the entire run to date, including intermediate means and event frequencies. Intermediate 10th–90th percentiles use 128-bin histograms, with exact zero mass; recovery percentiles use the complete sorted outcome sample. Node selection highlights direct dependency links and exposes sources, formulas, uncertainties and relevant assumptions. Zoom, a larger overlay and a fit-width control support navigation.

This is an exploratory scenario network, not a calibrated legal or fraud prediction. More trials reduce Monte Carlo noise; more nodes make assumptions explicit. Narrower ranges or larger concentrations express stronger analyst assumptions and do not establish greater accuracy. Conditional focus scores measure associations and are not causal effects or expected value of additional evidence. The medical/no-fault offset and the non-economic multiplier are scenario proxies requiring case-specific reconciliation. Fees and expenses are excluded.
