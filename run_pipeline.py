import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import clean
import load_data
import score
import train

STEPS = [("1. Load raw data into MySQL", load_data.main),
         ("2. Clean transactions", clean.main),
         ("3-4. Build features, train and validate models", train.main),
         ("5. Score customers and assign segments", score.main)]

if __name__ == "__main__":
    skip_load = "--skip-load" in sys.argv
    for name, step in STEPS:
        if skip_load and step is load_data.main:
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        step()
        print(f"--- done in {time.time() - t0:.1f}s")
