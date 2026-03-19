# DND-like Dataset Project

## Project Overview
This project is an automated pipeline for collecting and processing TTRPG (DND, Pathfinder, etc.) and fantasy lore for LLM fine-tuning.

## Guidelines
- **Scraping**: Use `scraper.py`. Target data goes to `data/raw/`.
- **Building**: Use `build_dataset.py`. Output to `data/finetune/`.
- **Validation**: Use `validate_dataset.py` to ensure high-quality tokens and language ratios.
- **Workflow**: `pipeline.py` is the main entry point for end-to-end tasks.

## Agents
This project uses specialized agents located in `.claude/agent/`:
- `dataset-scraper.md`: Handles web data acquisition and discovery.
- `dataset-builder.md`: Handles data formatting, conversion (Alpaca/ShareGPT), and validation.
- `dataset-ops.md`: Handles pipeline automation, scheduling, and dataset transfers.

## Coding Style
- Python: Use type hints, adhere to PEP8.
- Configuration: All logic should be driven by YAML configs (`scraper_config.yaml`, `transfer_config.yaml`).
