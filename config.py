import os
from pathlib import Path


def set_source_data_dirs() -> None:
    global source_dir
    global cases_dir
    global vocab_dir
    global output_dir

    source_dir = Path("datasets")
    cases_dir = source_dir / "cases"
    vocab_dir = source_dir / "dictionaries"
    output_dir = source_dir / "experiments"
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
