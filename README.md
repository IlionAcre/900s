# Nonprofit Filing Exporter

A GUI application for scraping and exporting IRS 990 and 990-PF nonprofit tax filings from ProPublica's Nonprofit Explorer.

## Features

- **Single or batch query mode**
- **Automatic filing type detection** (990 vs 990-PF)
- **Rate limit handling** with automatic retries
- **Stop button** to save partial progress
- **Session-per-batch** for better rate limit management
- **Live countdown timers** during waits
- **Interruptible waits** - stop immediately without waiting

## Requirements

- Python 3.10+
- uv package manager (or pip)

## Installation

### macOS / Linux

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
cd 900s

# Install dependencies
uv sync
```

### Windows

```powershell
# Install uv (if not already installed)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Navigate to the project directory
cd 900s

# Install dependencies
uv sync
```

## Usage

### Running the app

```bash
uv run main.py
```

### Single Query Mode

1. Select "Single query" mode
2. Enter an organization name
3. Click "Run"
4. The app will save results to `data/<org_name>_990.csv` or `data/<org_name>_990pf.csv`

### Multi Query Mode

1. Create a `queries.txt` file with one organization name per line (or use CSV format)
2. Select "Multi query" mode
3. Configure settings:
   - **Wait between queries**: Default 0s (increase if you get rate limited)
   - **Batch size**: 50 queries per batch
   - **Batch wait**: 61 seconds between batches (to avoid rate limits)
4. Click "Run"
5. Results saved to `data/results_990.csv` and `data/results_990pf.csv`

### Stop Button

Click "Stop" at any time to:
- Save all progress collected so far
- Cancel any current wait immediately
- Generate a summary report

## File Structure

### Core files (required)
- `main.py` - Main GUI application
- `scraper_unified.py` - Web scraping logic
- `column_selector.py` - Column filtering
- `process_990.py` - 990 form parser
- `process_990PF.py` - 990-PF form parser
- `process_historical_data.py` - Historical data parser
- `pyproject.toml` - Project dependencies
- `uv.lock` - Dependency lock file

### Configuration
- `.python-version` - Python version specification
- `queries.txt` - Example queries file

## Output

The app creates these files in the `data/` directory:
- `<name>_990.csv` - Cleaned 990 data
- `<name>_990pf.csv` - Cleaned 990-PF data
- `<name>_990_full.csv` - Full uncleaned 990 data (if "export full" is checked)
- `<name>_990pf_full.csv` - Full uncleaned 990-PF data (if "export full" is checked)
- `report_<date>.csv` - Processing report with status for each query

## Rate Limiting

The app includes several rate limit protections:
- Request-level retry (3 attempts with 61s delay)
- Batch-level session recreation
- Global rate limit gate (pauses all requests on 429)
- Configurable wait times
- Auto-stop after 2 consecutive 429s

## Troubleshooting

### "Rate limited" message
- Increase wait time between queries (try 5-10 seconds)
- Increase batch wait time
- Try again after 1 hour

### tkinter/GUI issues on macOS
If you get "no display name" errors, you may need to install Python with tkinter support:
```bash
brew install python-tk@3.13
```

### Dependencies not found
```bash
uv sync --reinstall
```

## License

MIT
