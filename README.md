# Streams Server

This repository contains the Vercel-deployable version of the Streams board game app.

## Included

- `server.py` Flask app
- `api/index.py` Vercel entrypoint
- `game.html` game UI
- `admin.html` record dashboard
- `vercel.json` routing config
- `requirements.txt` runtime dependencies
- `supabase_schema.sql` Supabase schema

## Local Run

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

## Test

```bash
python3 -m unittest discover -s tests
```

The test suite verifies that the API stays available in fallback AI mode when
PyTorch is not installed. This matches the default Vercel Python dependency set
in `requirements.txt`.

## Deployment

Deployments are triggered by pushing to the GitHub branch connected to Vercel.
Do not run `vercel deploy --prod` from this repository for the normal production
flow. GitHub Actions only runs validation; Vercel Git Integration is responsible
for build and deployment.

### Deployment Change

- Before: GitHub Actions installed Vercel CLI and ran `vercel deploy --prod`.
- Now: GitHub Actions only validates Python sources and fallback API behavior.
- Deployment trigger: Vercel Git Integration deploys after GitHub push/merge to
  the connected branch.
- Developer action: keep Vercel project connected to this GitHub repository and
  manage production environment variables in Vercel project settings.
- Developer action: do not add Vercel CLI production deploy steps back to GitHub
  Actions unless the deployment policy changes.

## Notes

- The app is linked to the Vercel project in `.vercel/project.json` locally.
- Supabase is used for persistent records.
- PyTorch is optional. If it is unavailable, `/api/ai_move` uses the fallback AI
  path and `/api/health` reports `ai_mode: "fallback"`.
