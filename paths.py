"""Where the current dataset is stored. Change it here to point the project at other data."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The 2,000 users with written personas, which experiments use.
DATA_DIR = ROOT / "data"
MAIN_DB = DATA_DIR / "main.db"
PERSONA_DB = DATA_DIR / "personas.db"

# The original 10,000 profiles with trait-score personas. New users are sampled from these.
SOURCE_MAIN_DB = ROOT / "main.db"
SOURCE_PERSONA_DB = ROOT / "personas.db"
