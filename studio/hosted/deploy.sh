#!/bin/sh
# Build the hosted site and deploy it to Vercel as the project "sapien".
# Run from the repo root: sh studio/hosted/deploy.sh
# SCOPE is the Vercel team. The site is served at sapien.trisanth.com.
set -e
SCOPE="${SAPIEN_VERCEL_SCOPE:-trisanthsrinis-projects}"
.venv/bin/python studio/hosted/build_site.py
cd studio/hosted/dist
npx -y vercel@latest link --yes --project sapien --scope "$SCOPE" >/dev/null
npx -y vercel@latest deploy --prod --yes --scope "$SCOPE"
