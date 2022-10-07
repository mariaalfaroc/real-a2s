import argparse, random


# Seed
random.seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Exchange part of the paths in the speficied file (random selection).")
    parser.add_argument("--file", type=str, required = True, help="File in which to apply the process")
    parser.add_argument("--perc", type=float, required = True, help="Percentage of files to exchange")
    parser.add_argument("--in_str", type=str, required = True, help="Source string")
    parser.add_argument("--out_str", type=str, required = True, help="Target string")
    args = parser.parse_args()

    return args


def exchange_names(args):

    # Open file:
    with open(args.file) as f:
        infile = f.read().splitlines()

    idx_list = sorted(random.sample(range(len(infile)), int(len(infile)*args.perc/100.0)))


    with open(args.file, 'w') as fout:
        for idx in range(len(infile)):
            line = infile[idx]

            if idx in idx_list: new_line = line.split(" ")[0].replace(args.in_str, args.out_str) + " " + line.split(" ")[1]
            else: new_line = line

            fout.write(new_line + '\n')

    return





if __name__ == '__main__':


    # /home/user/data/ICASSP-Data/midi_tenor_sax_audio/octave_transposed_
    # /home/user/data/ICASSP-Data/audiosWAV_tenor_sax/

    args = parse_arguments()
    # Print experiment details
    print(args)

    exchange_names(args)