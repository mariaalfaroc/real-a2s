import os
from pathlib import Path

# Dataset: SARA (JC DATASET) -> See DATASETS/SARA/info.txt
def set_source_data_dirs() -> None:
    global source_dir
    global cases_dir
    global vocab_dir
    global output_dir

    source_dir = Path(f"DATASETS/SARA")
    cases_dir = source_dir / "Cases"
    vocab_dir = source_dir / "dictionaries"
    output_dir = source_dir / "experiments"
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)