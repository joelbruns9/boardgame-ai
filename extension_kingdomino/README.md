# BGA Kingdomino Advisor

Experimental Board Game Arena overlay for the local Kingdomino advisor.

## Quick start

1. **Copy the best checkpoint** into the canonical location the server
   autodiscovers:

   ```text
   runs/kingdomino/best_checkpoint/current_best.pt
   ```

   Architecture (channels/blocks/bilinear_dim) is read from the checkpoint
   itself, so any model size works without further configuration.

2. **Start the server** from the repo root:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn games.kingdomino.web_app:app --host 127.0.0.1 --port 8000
   ```

3. **Load the extension** in Firefox (see below).

4. **Navigate to a live BGA Kingdomino game** on an active table during your
   `chooseDomino` or `placeDomino` turn.

5. **Open the popup** → set engine to **NN/MCTS**, leave **Checkpoint path
   blank** (the server uses `current_best.pt` automatically). Click
   **Capture Debug Only** first to confirm the scraper sees the page, then
   **Capture and Recommend**.

## Server

The extension posts to:

```text
http://127.0.0.1:8000/api/recommend
```

The server is the single source of truth for the model architecture: it reads
`channels`/`blocks`/`bilinear_dim` from the checkpoint's stored config and builds
the network to match. The extension does **not** send architecture — it only
sends an optional `checkpoint_path`. Leave the checkpoint path blank to use the
autodiscovered best model:

1. `runs/kingdomino/best_checkpoint/current_best.pt` (canonical best), else
2. the highest-iteration `iter_*.pt` found under `runs/kingdomino/`.

## Firefox (primary)

Minimum Firefox version: **111** (required for MAIN-world script injection).

Load this folder as a temporary extension:

```text
about:debugging#/runtime/this-firefox
```

Click **Load Temporary Add-on…** and select **`manifest.firefox.json`**.

## Chrome (secondary)

Load this folder from:

```text
chrome://extensions
```

Enable Developer mode, click **Load unpacked**, and select the folder. Chrome
uses **`manifest.json`** (MV3 service worker).

> Note: a browser only reads a file literally named `manifest.json`. To load the
> Firefox build in Chrome (or vice versa) you would copy the desired manifest
> over `manifest.json` in a temporary copy of this folder. Firefox's
> `about:debugging` lets you pick `manifest.firefox.json` directly.

## Current scraper status

The extension includes the overlay, popup, local-server request flow, and a
debug-first scraper. If it cannot normalize the live BGA page into the
Kingdomino engine state JSON yet, it will show a debug overlay and save the raw
capture in extension storage as `kingdomino_last_capture`.

Send that captured payload or the relevant BGA HTML when scraper normalization
needs to be tightened.
