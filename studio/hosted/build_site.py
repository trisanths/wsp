"""Build the hosted Sapien site for Vercel.

The hosted site is the web app in demo mode. It has no server: the mock API runs in the
browser, and the parts library is a snapshot of the real library, taken from this checkout
when the site is built. No keys and no backend go into the output.

Run from the repo root:

    .venv/bin/python studio/hosted/build_site.py

The output goes to studio/hosted/dist. Deploy it with deploy.sh in this folder.
"""

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEB = os.path.join(ROOT, "studio", "web")
DIST = os.path.join(HERE, "dist")

# The API paths that demo mode reads from a server, and the file that holds each one.
SNAPSHOTS = {
    "/api/catalog": "snap/catalog.json",
    "/api/library/recipes": "snap/recipes.json",
    "/api/library/findings": "snap/findings.json",
    "/api/library/stitches": "snap/stitches.json",
}

# Set before the app's modules load. demo-flag.js reads it and keeps demo mode on.
HOSTED_FLAG = '<script>window.SAPIEN_HOSTED = true;</script>'


def scrub(value):
    """Replace this checkout's absolute path with a repo-relative one, so no local path is published."""
    if isinstance(value, str):
        return value.replace(ROOT + os.sep, "").replace(ROOT, ".")
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def snapshot(client, path):
    response = client.get(path)
    if response.status_code != 200:
        raise SystemExit("GET " + path + " returned " + str(response.status_code))
    return scrub(response.json())


def vercel_config():
    rewrites = [{"source": path, "destination": "/" + target} for path, target in SNAPSHOTS.items()]
    return {
        "cleanUrls": False,
        "trailingSlash": False,
        "rewrites": rewrites,
        "headers": [
            {"source": "/snap/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]},
        ],
    }


def main():
    sys.path.insert(0, ROOT)
    # The offline backend: the snapshot must never start a GPU job or reach Modal.
    os.environ["WSP_STUDIO_BACKEND"] = "stub"
    from fastapi.testclient import TestClient
    from studio.server.app import app

    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    shutil.copytree(WEB, DIST, ignore=shutil.ignore_patterns("*.md", ".DS_Store"))

    client = TestClient(app)
    for path, target in SNAPSHOTS.items():
        out = os.path.join(DIST, target)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as handle:
            json.dump(snapshot(client, path), handle, separators=(",", ":"))

    index = os.path.join(DIST, "index.html")
    with open(index) as handle:
        text = handle.read()
    if "<head>" not in text:
        raise SystemExit("index.html has no <head> tag")
    text = text.replace("<head>", "<head>\n  " + HOSTED_FLAG, 1)
    with open(index, "w") as handle:
        handle.write(text)

    with open(os.path.join(DIST, "vercel.json"), "w") as handle:
        json.dump(vercel_config(), handle, indent=2)

    catalog = json.load(open(os.path.join(DIST, "snap/catalog.json")))
    counts = catalog.get("counts", {})
    print("built " + DIST)
    print("parts " + str(counts.get("total")) + ", placeable " + str(counts.get("placeable")))


if __name__ == "__main__":
    main()
