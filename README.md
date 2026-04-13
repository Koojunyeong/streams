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
python server.py
```

## Notes

- The app is linked to the Vercel project in `.vercel/project.json` locally.
- Supabase is used for persistent records.
