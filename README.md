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

## Play Log Analysis

Generate the behavior-analysis artifacts from the saved `players / games / turns` data:

```bash
python streams_data_analysis/generate_play_log_report.py
```

Outputs are written to [streams_data_analysis/output](C:/Users/asus/Desktop/Streams_server/streams-server-deploy/streams_data_analysis/output):

- `game_level_metrics.csv`
- `turn_level_metrics.csv`
- `group_summary.csv`
- `play_log_analysis_report.md`
- `play_log_analysis_report.html`

The HTML report is the easiest artifact to review in a browser.
