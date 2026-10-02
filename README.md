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
