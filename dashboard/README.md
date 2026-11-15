# Swarm dashboard

A static page that shows the part 3 environment: the race between the agent swarm and the
equal-budget baselines, the agent log, a program inspector, fit against score, budget use,
the real Modal runs and the pilot parts. Every number keeps its labels (STUB, SEARCH,
HELD-OUT, INTERIM, SMOKE) from the rows it came from.

- `public/index.html`, `public/app.js`, `public/styles.css`: the page. No build step, no libraries.
- `public/data/*.json`: written by `scripts/export_dashboard.py` from `outputs/` and `data/`.
- `wrangler.jsonc`: a static-assets Worker named `wsp-swarm-dashboard` that serves `public/`.

## Refresh and deploy

```bash
bash scripts/deploy_dashboard.sh
```

This runs the export and then `npx --yes wrangler@4 deploy` from `dashboard/`. To check the page
before deploying:

```bash
bash scripts/deploy_dashboard.sh --no-deploy
.venv/Scripts/python -m http.server 8765 --directory dashboard/public
```

The export keeps the data under about 3 MB. When a new run pushes it over, it trims in a fixed
order (shorter heal curves, then a sample of low-fidelity log rows, then a cap per run) and the
page footer lists what it trimmed. It refuses to write if it finds a `.env` value, a token, a
local path or an email, because the deployed site is public.
