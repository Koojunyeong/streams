# Streams Data Analysis

This folder bundles the play-log behavior analysis workflow and its generated artifacts.

## Run

```bash
python streams_data_analysis/generate_play_log_report.py
```

## Output

Generated files are written to:

- `streams_data_analysis/output/game_level_metrics.csv`
- `streams_data_analysis/output/turn_level_metrics.csv`
- `streams_data_analysis/output/group_summary.csv`
- `streams_data_analysis/output/play_log_analysis_report.md`
- `streams_data_analysis/output/play_log_analysis_report.html`

The HTML file is the easiest version to open in a browser for presentation or review.
