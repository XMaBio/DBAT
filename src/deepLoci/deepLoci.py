#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import sys

from deepLoci.config import load_config
from deepLoci.data.data import process_data
from deepLoci.bin.train import train_predict



def validate_config(config_path):

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    return config



def print_custom_help():
    help_text = """
usage: dbat deepLoci --input <in_file> --output <out_dir> [options]

required argument:
     --input          input data file (csv format) 
     --output         output directory

options:
     --patience       early stop patience   (default: 3)
     --inilr          initial learning rate (default: 5e-5)
     --minlr          minimum learning rate (default: 5e-7)
     --workers        number of workers     (default: 4)

"""
    print(help_text)
    sys.exit(0)


    
def parse_arguments():
    # Load config for defaults
    deepLoci_dir = Path(__file__).resolve().parent
    config_path = deepLoci_dir / "config" / "cfg.yml"
    config = validate_config(config_path)
    input_params = config["input_params"]


    # Create parser WITHOUT default help
    parser = argparse.ArgumentParser(
        description="Predict sequencing peaks from DNA sequence",
        add_help=False  # disable default -h/--help
    )

    # Manual help
    parser.add_argument('--help', '-h', action='store_true',
                        help='Show help message and exit')

    # Required arg --input
    parser.add_argument('--input',    type=str, help=argparse.SUPPRESS)
    parser.add_argument('--output',   type=str, help=argparse.SUPPRESS)

    # Optional args
    parser.add_argument('--patience', type=str, default=input_params["patience"])
    parser.add_argument('--inilr',    type=str, default=input_params["inilr"])
    parser.add_argument('--minlr',    type=str, default=input_params["minlr"])
    parser.add_argument('--workers',  type=str, default=input_params["workers"])

    args = parser.parse_args()

    if args.help:
        print_custom_help()

    # Require --input
    if args.input is None:
        print("error: the following arguments are required: --input", file=sys.stderr)
        print_custom_help()

    if args.output is None:
        print("error: the following arguments are required: --output", file=sys.stderr)
        print_custom_help()

    return args, config



def run_deepLoci():
    args, config = parse_arguments()

    input_file = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    patience   = args.patience
    ini_lr     = args.inilr
    min_lr     = args.minlr
    workers    = args.workers

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("DBAT deepLoci - Prediction of Featured Loci.")
    print("=" * 70)

    print(f" Input parameters:")
    print(f"   - input file: {input_file}")
    print(f"   - output dir: {output_dir}")
    print(f"   - early stop patience: {patience}")
    print(f"   - initial learning rate: {ini_lr}")
    print(f"   - minimum learning rate: {min_lr}")
    print(f"   - number of workers: {workers}")

    print("=" * 70)

    # process data file
    data_types = process_data(str(input_file), str(output_dir))

    # Convert parameters to correct types before passing to train_predict
    patience = int(patience)
    ini_lr   = float(ini_lr)
    min_lr   = float(min_lr)
    workers  = int(workers)

    # train and prediction
    train_predict(data_types, str(output_dir), patience, ini_lr, min_lr, workers)

if __name__ == "__main__":
    run_deepLoci()
