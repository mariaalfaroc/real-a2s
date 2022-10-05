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
    parser.add_argument("--train_ft", type=str, required = False, help="Fine-tuning train data partition")
    parser.add_argument("--val_ft", type=str, required = False, help="Fine-tuning validation data partition")
    parser.add_argument("--test_ft", type=str, required = False, help="Fine-tuning test data partition")
    parser.add_argument("--trainmodel", type=str2bool, default="True", help="Whether to initially train the model")
    parser.add_argument("--finetune", type=str2bool, default="False", help="Whether to finetune the model")
    parser.add_argument("--freeze", type=str, required=False, help="Layers to update when freezing the model", default=None)
    args = parser.parse_args()


    if args.freeze is not None: args.freeze = [item for item in args.freeze.split(',')]
    
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

    # Pretrain
    if args.trainmodel == True: 
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




    # Fine tune the model:
    if args.finetune == True:
        # Checking that pretrained model exists:
        assert os.path.exists(model_filepath), 'Model does not exist'
        model.load(model_filepath, map_location=device)
        print("Loaded pretrained model from {}".format(model_filepath))
        # summary(model.model)

        # Filepaths globals
        case = args.train_ft.split("_")[0]
        output_dir = config.output_dir / f"{case}_{args.encoding}"
        os.makedirs(output_dir, exist_ok=True)
        nameOfVoc = "Vocab"
        nameOfVoc = nameOfVoc + "_woutmultirest" if not args.multirest else nameOfVoc
        nameOfVoc = nameOfVoc + "_" + args.train.split("-")[0] + 'FT'
        nameOfVoc = nameOfVoc + "_" + args.encoding
        multirest_appedix = "_withmultirest" if args.multirest else ''
        freeze_appendix = '_update' + "".join([u.capitalize() for u in args.freeze]) if args.freeze is not None else '_updateALL'
        model_filepath = output_dir / f"model{multirest_appedix}{freeze_appendix}.pt"
        logs_path = output_dir / f"results{multirest_appedix}{freeze_appendix}.csv"

        # Loading data:
        XFTrain_ft, YFTrain_ft, XFVal_ft, YFVal_ft, XFTest_ft, YFTest_ft = load_data_from_files(config.cases_dir / args.train_ft,\
            config.cases_dir / args.val_ft, config.cases_dir / args.test_ft, args.multirest)

        # Dictionaries:
        w2i_ft, i2w_ft = check_and_retrieveVocabulary_from_files(nameOfVoc=nameOfVoc, multirest=args.multirest, encoding=args.encoding,\
            YTrain = YFTrain_ft, YVal = YFVal_ft, YTest = YFTest_ft)


        # Changing the size of the output:
        model.updateCRNNOutput(w2i_ft, i2w_ft)

        # Freezing the model except for the specified parts:
        if args.freeze is not None: model.freezeModel(list_update_elements=args.freeze)


        # Destination paths:
        modelft_filepath = model_filepath.parent / model_filepath.name.replace('.pt', '_ft.pt')
        logs_path = logs_path.parent / logs_path.name.replace('.pt', '_ft.pt')
        
        # Fine-tune, validate, and test the model:
        test_ser = model.fit(
            train_data_generator(
                XFiles=XFTrain_ft, YFiles=YFTrain_ft,
                batch_size=args.batch_size,
                width_reduction=model.model.encoder.width_reduction,
                w2i=w2i_ft,
                device=device,
                encoding=args.encoding
            ),
            epochs=args.epochs,
            steps_per_epoch=len(XFTrain_ft) // args.batch_size,
            val_data=(XFVal_ft, YFVal_ft),
            test_data=(XFTest_ft, YFTest_ft),
            batch_size=args.batch_size,
            patience=args.patience,
            weights_path=model_filepath,
            logs_path=logs_path
        )
    pass

if __name__ == "__main__":
    main()
