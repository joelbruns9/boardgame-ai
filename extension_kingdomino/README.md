# BGA Kingdomino Advisor

Experimental Board Game Arena overlay for the local Kingdomino advisor.

## Server

Start the local server from the repo root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn games.kingdomino.web_app:app --host 127.0.0.1 --port 8000
```

The extension posts to:

```text
http://127.0.0.1:8000/api/recommend
```

## Firefox

Load this folder as a temporary extension:

```text
about:debugging#/runtime/this-firefox
```

Use `manifest.json`.

## Chrome

Chrome Manifest V3 expects a background service worker. To test in Chrome,
copy `manifest.chrome.json` over `manifest.json` in a temporary copy of this
folder, then load the folder from:

```text
chrome://extensions
```

## Current Scraper Status

The extension includes the overlay, popup, local-server request flow, and a
debug-first scraper. If it cannot normalize the live BGA page into the
Kingdomino engine state JSON yet, it will show a debug overlay and save the raw
capture in extension storage as `kingdomino_last_capture`.

Send that captured payload or the relevant BGA HTML when scraper normalization
needs to be tightened.
