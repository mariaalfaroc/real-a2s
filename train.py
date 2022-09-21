import os, argparse, gc, random

import torch
import numpy as np

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import config
from data import load_data, check_and_retrieveVocabulary, train_data_generator
from model import CTCTrainedCRNN
from utils import write_plot_results

def str2bool(v: str) -> bool:
    if v == "True":
        return True
    return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Supervised training arguments.")
    parser.add_argument("--dataset", type=str, default="Primus", choices=["Primus","Sax"], help="Name of the dataset to use")
    parser.add_argument("--num_samples", type=int, required=True, help="Dataset size")
    parser.add_argument("--multirest", type=str2bool, default="False", help="Whether to use samples that contain multirest")
    parser.add_argument("--encoding", type=str, default="kern", choices=["kern","decoupled"], help="Encoding type to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--num_iters", type=int, default=5, help="Number of complete training iterations")
    args = parser.parse_args()
    return args

def main():
    gc.collect()
    torch.cuda.empty_cache()

    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Device {device}")

    args = parse_arguments()
    # Print experiment details
    print("Supervised training experiment")
    print(args)

    # Data globals
    config.set_source_data_dirs(source_path=args.dataset)
    print(f"Data used {config.source_dir.stem}")
    output_dir = config.output_dir / f"{args.num_samples}"
    os.makedirs(output_dir, exist_ok=True)
    nameOfVoc = "Vocab"
    nameOfVoc = nameOfVoc + "_woutmultirest" if not args.multirest else nameOfVoc
    nameOfVoc = nameOfVoc + "_" + args.encoding

    # k-fold experiment
    results = []
    for num_iter in range(args.num_iters):
        print(f"Iter {num_iter}")

        # Set filepaths outputs
        name = f"Iter{num_iter}"
        name = name + "_withmultirest" if args.multirest else name
        model_filepath = output_dir / f"{name}.pt"
        logs_path = output_dir / f"{name}.csv"

        # Data
        # 60% - 20% - 20% -> Partitions depend on the number of iteration
        XFTrain, YFTrain, XFVal, YFVal, XFTest, YFTest = load_data(num_samples=args.num_samples, num_iter=num_iter, multirest=args.multirest)
        w2i, i2w = check_and_retrieveVocabulary(nameOfVoc=nameOfVoc, multirest=args.multirest, encoding=args.encoding)
        
        # Model
        model = CTCTrainedCRNN(dictionaries=(w2i, i2w), encoding=args.encoding, device=device)

        # Train, validate, and test
        test_ser = model.fit(
            train_data_generator(
                XFiles=XFTrain, YFiles=YFTrain,
                batch_size=args.batch_size,
                width_reduction=model.model.encoder.width_reduction,
                w2i=w2i,
                device=device,
                encoding=args.encoding
            ),
            epochs=args.epochs,
            steps_per_epoch=len(XFTrain) // args.batch_size,
            val_data=(XFVal, YFVal),
            test_data=(XFTest, YFTest),
            batch_size=args.batch_size,
            patience=args.patience,
            weights_path=model_filepath,
            logs_path=logs_path
        )
        results.append(test_ser)
        del model
    
    # Compute cross-validation results and save them
    filename = f"{args.num_iters}cv"
    filename = filename + "_withmultirest" if args.multirest else filename
    filename += "_results.dat"
    write_plot_results(plotfile=output_dir / filename, results=results)
    
    pass

if __name__ == "__main__":
    main()
