#!/usr/bin/env python3
"""
Sync soccer_team_app_standalone_backup.html from soccer_team_app.jsx.

Keeps the HTML scaffolding (Firebase setup, inline icon library, render bootstrap)
and replaces the app code section between the BEGINS/ENDS markers with the JSX
contents, adapted as follows:
  - drop the `import React, ...` line
  - drop the localStorage-based storageGet/storageSet helpers
  - change `export default function App()` -> `function App()`
  - replace the App component's data-loading useEffect with a Firestore listener
  - replace persistRoster/persistGames with Firestore-backed versions
  - inject TEAM_DOC_ID / teamDoc / gamesCol helpers
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

# CLI flags: --preview (draft deploy) or --prod (promote to production).
DEPLOY_MODE = None
if "--prod" in sys.argv:
    DEPLOY_MODE = "prod"
elif "--preview" in sys.argv:
    DEPLOY_MODE = "preview"
JSX = HERE / "soccer_team_app.jsx"
HTML = HERE / "soccer_team_app_standalone_backup.html"

jsx_src = JSX.read_text()
html_src = HTML.read_text()

# 1. Drop the import line.
jsx_body = re.sub(
    r"^import React,.*?from 'lucide-react';\n\n",
    "",
    jsx_src,
    count=1,
    flags=re.S,
)

# 2. Drop the storageGet / storageSet helpers (and the STORAGE_KEYS const).
jsx_body = re.sub(
    r"const STORAGE_KEYS = \{[^}]*\};\n\n",
    "",
    jsx_body,
    count=1,
)
jsx_body = re.sub(
    r"async function storageGet\(key\) \{.*?\n\}\n\n"
    r"async function storageSet\(key, value\) \{.*?\n\}\n\n",
    "",
    jsx_body,
    count=1,
    flags=re.S,
)

# 3. Inject Firestore helpers right before formatClock.
firestore_helpers = (
    "// ----- Firestore data layer -----\n"
    "// Single shared team — everyone with the URL sees the same data.\n"
    "const TEAM_DOC_ID = 'main';\n"
    "function teamDoc() { return window.fbDb.collection('teams').doc(TEAM_DOC_ID); }\n"
    "function gamesCol() { return teamDoc().collection('games'); }\n\n"
)
jsx_body = jsx_body.replace(
    "function formatClock(seconds) {",
    firestore_helpers + "function formatClock(seconds) {",
    1,
)

# 4. export default function App() -> function App()
jsx_body = jsx_body.replace(
    "export default function App() {",
    "function App() {",
    1,
)

# 5. Replace the data-loading useEffect with Firestore listeners.
old_load_effect = (
    "  useEffect(() => {\n"
    "    (async () => {\n"
    "      let loadedRoster = null;\n"
    "      try {\n"
    "        const r = await storageGet(STORAGE_KEYS.ROSTER);\n"
    "        if (r?.value) {\n"
    "          const parsed = JSON.parse(r.value);\n"
    "          if (Array.isArray(parsed) && parsed.length > 0) loadedRoster = parsed;\n"
    "        }\n"
    "      } catch (e) {}\n"
    "      if (!loadedRoster) {\n"
    "        loadedRoster = SEED_ROSTER;\n"
    "        try { await storageSet(STORAGE_KEYS.ROSTER, JSON.stringify(SEED_ROSTER)); } catch (e) {}\n"
    "      }\n"
    "      setRoster(loadedRoster);\n\n"
    "      try {\n"
    "        const g = await storageGet(STORAGE_KEYS.GAMES);\n"
    "        if (g?.value) setGames(JSON.parse(g.value));\n"
    "      } catch (e) {}\n"
    "      try {\n"
    "        const w = await storageGet(STORAGE_KEYS.WEIGHTS);\n"
    "        if (w?.value) setWeights(mergeWeights(JSON.parse(w.value)));\n"
    "      } catch (e) {}\n"
    "      setLoaded(true);\n"
    "    })();\n"
    "  }, []);"
)
new_load_effect = (
    "  // Subscribe to Firestore on mount (after auth ready)\n"
    "  useEffect(() => {\n"
    "    let unsubRoster = null;\n"
    "    let unsubGames = null;\n\n"
    "    window.fbReady.then((ok) => {\n"
    "      if (!ok) return;\n\n"
    "      // Roster + weights listener (both live on the team doc)\n"
    "      unsubRoster = teamDoc().onSnapshot((snap) => {\n"
    "        if (snap.exists) {\n"
    "          const data = snap.data();\n"
    "          if (Array.isArray(data.roster) && data.roster.length > 0) {\n"
    "            setRoster(data.roster);\n"
    "          } else {\n"
    "            teamDoc().set({ roster: SEED_ROSTER }, { merge: true });\n"
    "          }\n"
    "          if (data.weights) setWeights(mergeWeights(data.weights));\n"
    "        } else {\n"
    "          teamDoc().set({ roster: SEED_ROSTER });\n"
    "        }\n"
    "        setLoaded(true);\n"
    "      }, (err) => {\n"
    "        console.error('Roster listener error:', err);\n"
    "        setLoaded(true);\n"
    "      });\n\n"
    "      // Games listener (sorted by date desc client-side)\n"
    "      unsubGames = gamesCol().onSnapshot((snap) => {\n"
    "        const fetched = snap.docs.map((d) => ({ ...d.data(), id: d.id }));\n"
    "        fetched.sort((a, b) => new Date(b.date) - new Date(a.date));\n"
    "        setGames(fetched);\n"
    "      }, (err) => {\n"
    "        console.error('Games listener error:', err);\n"
    "      });\n"
    "    });\n\n"
    "    return () => {\n"
    "      if (unsubRoster) unsubRoster();\n"
    "      if (unsubGames) unsubGames();\n"
    "    };\n"
    "  }, []);"
)
if old_load_effect not in jsx_body:
    raise SystemExit("Could not find original load useEffect block to replace.")
jsx_body = jsx_body.replace(old_load_effect, new_load_effect, 1)

# 6. Replace persistRoster / persistGames with Firestore versions.
old_persist = (
    "  const persistRoster = async (next) => {\n"
    "    setRoster(next);\n"
    "    try { await storageSet(STORAGE_KEYS.ROSTER, JSON.stringify(next)); } catch (e) {}\n"
    "  };\n\n"
    "  const persistGames = async (next) => {\n"
    "    setGames(next);\n"
    "    try { await storageSet(STORAGE_KEYS.GAMES, JSON.stringify(next)); } catch (e) {}\n"
    "  };\n\n"
    "  const persistWeights = async (next) => {\n"
    "    const merged = mergeWeights(next);\n"
    "    setWeights(merged);\n"
    "    try { await storageSet(STORAGE_KEYS.WEIGHTS, JSON.stringify(merged)); } catch (e) {}\n"
    "  };"
)
new_persist = (
    "  const persistRoster = async (next) => {\n"
    "    setRoster(next); // optimistic\n"
    "    try { await teamDoc().set({ roster: next }, { merge: true }); } catch (e) { console.error(e); }\n"
    "  };\n\n"
    "  const persistGames = async (next) => {\n"
    "    // Diff against current and write only what changed.\n"
    "    const prevById = Object.fromEntries(games.map(g => [g.id, g]));\n"
    "    const nextById = Object.fromEntries(next.map(g => [g.id, g]));\n"
    "    setGames(next); // optimistic\n"
    "    const writes = [];\n"
    "    for (const g of next) {\n"
    "      const prev = prevById[g.id];\n"
    "      if (!prev || JSON.stringify(prev) !== JSON.stringify(g)) {\n"
    "        const { id, ...payload } = g;\n"
    "        writes.push(gamesCol().doc(id).set(payload));\n"
    "      }\n"
    "    }\n"
    "    for (const id of Object.keys(prevById)) {\n"
    "      if (!nextById[id]) writes.push(gamesCol().doc(id).delete());\n"
    "    }\n"
    "    try { await Promise.all(writes); } catch (e) { console.error('Save error:', e); }\n"
    "  };\n\n"
    "  const persistWeights = async (next) => {\n"
    "    const merged = mergeWeights(next);\n"
    "    setWeights(merged); // optimistic\n"
    "    try { await teamDoc().set({ weights: merged }, { merge: true }); } catch (e) { console.error('Weights save error:', e); }\n"
    "  };"
)
if old_persist not in jsx_body:
    raise SystemExit("Could not find original persistRoster/persistGames to replace.")
jsx_body = jsx_body.replace(old_persist, new_persist, 1)

# Splice into HTML between markers.
begin_marker = "// ===== APP CODE BEGINS ====="
end_marker = "// ===== APP CODE ENDS ====="
pattern = re.compile(
    r"(" + re.escape(begin_marker) + r"\n)(.*?)(\n\s*" + re.escape(end_marker) + r")",
    re.S,
)

# Indent? The original app code lives inside a <script> block but is not indented
# (lines start at column 0). Keep that convention.
new_html = pattern.sub(
    lambda m: m.group(1) + "\n" + jsx_body + "\n" + m.group(3),
    html_src,
    count=1,
)

if new_html == html_src:
    raise SystemExit("HTML splice failed: markers not found or content unchanged.")

HTML.write_text(new_html)
print(f"Wrote {HTML} ({len(new_html.splitlines())} lines)")

# Also build a Netlify-Drop-ready folder: a SINGLE index.html with every static
# asset inlined as a base64 data URI. One-file deploys = no separate asset
# uploads on Netlify Drop (saves bandwidth / build minutes when only the HTML
# actually changed).
#
# Exception: PWA support files (manifest, service worker, install icons) MUST
# remain as real files in the deploy folder so the browser can fetch them by
# URL — data URIs don't work for these. They are copied as-is alongside the
# inlined index.html.
import shutil
import base64
import mimetypes

DEPLOY_DIR = Path.home() / "Desktop" / "stompers_deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# Files that must stay as real files in the deploy folder (PWA shell).
PWA_FILES = [
    "manifest.webmanifest",
    "sw.js",
    "icon-192.png",
    "icon-512.png",
    "icon-maskable-192.png",
    "icon-maskable-512.png",
]

# Inline every "./asset" reference in src=/href= as base64, EXCEPT PWA shell
# files which need to remain fetchable by URL. Files that don't exist locally
# are left untouched (so external URLs still work).
asset_refs = set(re.findall(r'(?:src|href)="\./([^"?#]+)"', new_html))
deploy_html = new_html
inlined = []
for name in sorted(asset_refs):
    if name in PWA_FILES:
        continue  # keep as real file reference
    src = HERE / name
    if not src.is_file():
        continue
    mime, _ = mimetypes.guess_type(str(src))
    if not mime:
        mime = "application/octet-stream"
    b64 = base64.b64encode(src.read_bytes()).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    deploy_html = deploy_html.replace(f'"./{name}"', f'"{data_uri}"')
    inlined.append(f"{name} ({len(b64)//1024} KB b64)")

# Files that must exist in the deploy folder forever (never purged by stale
# cleanup). index.html + every PWA support file.
KEEP_IN_DEPLOY = {"index.html", *PWA_FILES}

# Remove anything else left over from previous deploys.
for stale in DEPLOY_DIR.iterdir():
    if stale.name in KEEP_IN_DEPLOY:
        continue
    if stale.is_file():
        stale.unlink()
    else:
        shutil.rmtree(stale)

(DEPLOY_DIR / "index.html").write_text(deploy_html)

# Copy PWA shell files into the deploy folder.
copied_pwa = []
for name in PWA_FILES:
    src = HERE / name
    if not src.is_file():
        print(f"  WARNING: PWA file missing: {name}")
        continue
    shutil.copy2(src, DEPLOY_DIR / name)
    copied_pwa.append(name)

print(f"Built {DEPLOY_DIR} (index.html {len(deploy_html)//1024} KB + {len(copied_pwa)} PWA files)")
if inlined:
    print(f"  Inlined: {', '.join(inlined)}")
if copied_pwa:
    print(f"  PWA shell: {', '.join(copied_pwa)}")
print("Drag the FOLDER (not just index.html) onto Netlify Drop so PWA install works.")

# Optional: invoke Netlify CLI for one-step deploys.
#   python3 _sync_html.py --preview   → draft URL, production untouched
#   python3 _sync_html.py --prod      → promote to lasalle-stompers.netlify.app
if DEPLOY_MODE:
    import subprocess
    cmd = ["netlify", "deploy", "--dir", str(DEPLOY_DIR)]
    if DEPLOY_MODE == "prod":
        cmd.append("--prod")
        print(f"\n>>> Deploying to PRODUCTION: {' '.join(cmd)}")
    else:
        print(f"\n>>> Deploying DRAFT preview: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=DEPLOY_DIR, check=True)
    except FileNotFoundError:
        print("ERROR: netlify CLI not found. Install with: brew install netlify-cli")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: netlify deploy failed (exit {e.returncode})")
        sys.exit(e.returncode)

