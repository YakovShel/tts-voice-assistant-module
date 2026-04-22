from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / 'templates'
LOGS_DIR = BASE_DIR / 'logs'
CACHE_DIR = BASE_DIR / 'cache'
OUTPUT_DIR = BASE_DIR / 'generated_audio'
HISTORY_FILE = LOGS_DIR / 'history.jsonl'

for directory in (TEMPLATES_DIR, LOGS_DIR, CACHE_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '5000'))
DEFAULT_SAMPLE_RATE = int(os.getenv('DEFAULT_SAMPLE_RATE', '24000'))
