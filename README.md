# Scarping & Analysis Project

This project is structured for sentiment analysis tasks.

## Project Structure

```
├── .vscode/              # VS Code settings
├── .github/              # GitHub Actions workflows
├── src/                  # Source code
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
└── scripts/             # Utility scripts
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development

- Use `src/` for main implementation
- Use `notebooks/` for exploration and analysis
- Write tests in `tests/`
- Add utility scripts to `scripts/`

## Testing

Run tests using pytest:

```bash
python -m pytest tests/
```
