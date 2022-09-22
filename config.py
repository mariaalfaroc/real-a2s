import os, pathlib

encoding_options = ['kern', 'decoupled', 'decoupled_dot']


# Dataset available: PrIMuS
# It follows the structure set in set_data_dirs()
def set_source_data_dirs(source_path: str):
    global audio_extn
    global label_extn
    
    global source_dir
    global audios_dir
    global labels_dir
    global vocab_dir
    global output_dir

    source_dir = pathlib.Path(f"DATASETS/{source_path}")
    audios_dir = source_dir / "wav"
    labels_dir = source_dir / "krn"
    vocab_dir = source_dir / "dictionaries"
    output_dir = source_dir / "experiments"
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    audio_extn = ".wav"
    label_extn = ".krn"