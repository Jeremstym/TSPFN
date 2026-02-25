import sys
from pathlib import Path

# Find the project root (the directory containing 'tspfn' and 'submodules')
# Since we are in tspfn/foundationals/labram.py, we go up THREE levels
# OR we look for the presence of the 'submodules' folder
def find_project_root(current_path):
    for parent in current_path.parents:
        if (parent / "submodules").exists():
            return parent
    return current_path.parents[2] # Fallback

ROOT = find_project_root(Path(__file__).resolve())
LABRAM_PATH = ROOT / "submodules" / "labram"

if LABRAM_PATH.exists():
    if str(LABRAM_PATH) not in sys.path:
        sys.path.insert(0, str(LABRAM_PATH)) # Use insert(0) to prioritize this path
    print(f"Successfully added LaBraM path: {LABRAM_PATH}")
else:
    print(f"Warning: LaBraM submodule not found at {LABRAM_PATH}")

from modeling_vqnsp import vqnsp_encoder_base_decoder_3x200x12