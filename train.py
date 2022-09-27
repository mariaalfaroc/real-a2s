import os, argparse, gc, random, config

import torch
import numpy as np

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import config
from data import train_data_generator, load_data_from_files, check_and_retrieveVocabulary_from_files
from model import CTCTrainedCRNN
from utils import write_plot_results

def str2bool(v: str) -> bool:
    if v == "True":
        return True
    return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Supervised training arguments.")
    parser.add_argument("--dataset", type=str, default="Primus", choices=["Primus","SARA"], help="Name of the dataset to use")
    parser.add_argument("--multirest", type=str2bool, default="False", help="Whether to use samples that contain multirest")
    parser.add_argument("--encoding", type=str, default="kern", choices=config.encoding_options, help="Encoding type to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--train", type=str, required = True, help="Train data partition")
    parser.add_argument("--val", type=str, required = True, help="Validation data partition")
    parser.add_argument("--test", type=str, required = True, help="Test data partition")
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
    case = args.train.split("_")[0]
    output_dir = config.output_dir / f"{case}_{args.encoding}"
    os.makedirs(output_dir, exist_ok=True)
    nameOfVoc = "Vocab"
    nameOfVoc = nameOfVoc + "_woutmultirest" if not args.multirest else nameOfVoc
    nameOfVoc = nameOfVoc + "_" + args.train.split("-")[0]
    nameOfVoc = nameOfVoc + "_" + args.encoding


    # k-fold experiment
    results = []

    # Set filepaths outputs
    multirest_appedix = "_withmultirest" if args.multirest else ''
    model_filepath = output_dir / f"model{multirest_appedix}.pt"
    logs_path = output_dir / f"results{multirest_appedix}.csv"

    # Data
    XFTrain, YFTrain, XFVal, YFVal, XFTest, YFTest = load_data_from_files(config.cases_dir / args.train,\
        config.cases_dir / args.val, config.cases_dir / args.test, args.multirest)
    w2i, i2w = check_and_retrieveVocabulary_from_files(nameOfVoc=nameOfVoc, multirest=args.multirest, encoding=args.encoding,\
        YTrain = YFTrain, YVal = YFVal, YTest = YFTest)
    
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
    
    # # Compute cross-validation results and save them
    # filename = f"{args.num_iters}cv"
    # filename = filename + "_withmultirest" if args.multirest else filename
    # filename += "_results.dat"
    # write_plot_results(plotfile=output_dir / filename, results=results)
    
    pass

if __name__ == "__main__":
    main()
