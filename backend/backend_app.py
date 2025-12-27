import os
import sys

# Ensure the project root (one level up from this backend folder) is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the FastAPI app from the main API package
from api.app import app  # noqa: E402

