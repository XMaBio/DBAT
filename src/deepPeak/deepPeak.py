#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import sys

from deepPeak.config import load_config
from deepPeak.bin import norm_train
from deepPeak.bin import norm_prediction

from deepPeak.bin import cross_train
from deepPeak.bin import cross_prediction


def validate_config(config_path):
    config = load_config(str(config_path))

    return config



def print_custom_help():
    help_text = """

Usage: dbat deepPeak --mode <MODE> --action <ACTION> [OPTIONS]

<MODE>
  norm        Train or predict on a single species.
  cross_spe   Train on ref species and predict on target species.
  
<ACTION>
  depend on mode

──────────────────────────────────────────────────────
Normal Mode (--mode norm)
──────────────────────────────────────────────────────
1. Training:
  --action train       
    --genome         Genome FASTA file           (required)
    --input          Training bigWig file        (required)
    --output         Output directory    (default: results)
    --suppress       Suppressive data in bigWig format, 
                     allow multiple times        (optional)

2. Prediction:
  --action prediction       
    --model          Trained model file          (required)
    --stats          Statistics file             (required)
    --genome         Genome FASTA file           (required)
    --output         Output directory    (default: results)

──────────────────────────────────────────────────────
Cross-Species Mode (--mode cross_spe)
──────────────────────────────────────────────────────
1. Train reference species:
  --action train       
    --genome        Ref species genome file      (required)
    --ref_wt        Ref species wt bigWig file   (required)
    --ref_mut       Ref species mut bigWig file  (required)
    --output        Output directory     (default: results)

2. Predict target species:
  --action prediction       
    --model        Trained model file            (required)
    --stats        Statistics file               (required)
    --genome       Target species genome file    (required)
    --targ_wt      Target species wt bigWig file (required)
    --output       Output directory      (default: results)

"""
    print(help_text)
    sys.exit(0)


def parse_arguments():
    # Load config for defaults
    deepPeak_dir = Path(__file__).resolve().parent
    config_path = deepPeak_dir / "config" / "cfg.yml"
    config = validate_config(config_path)
    input_params = config["input_params"]

    # Create parser WITHOUT default help
    parser = argparse.ArgumentParser(
        prog="dbat deepPeak",
        description="Predict sequencing peaks from DNA sequence",
        add_help=False  # disable default -h/--help
    )

    # Manual help
    parser.add_argument('--help', '-h', action='store_true',
                        help='Show help message and exit')

    # Required action
    parser.add_argument('--mode', type=str,
                        choices=['norm', 'cross_spe'],
                        help='Action to perform: "norm" or "cross_spe"')

    parser.add_argument('--action', type=str,
                        choices=['train', 'prediction'],
                        help='Action to perform: "train" or "prediction"')

    # Required args
    parser.add_argument('--genome', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--input',  type=str, help=argparse.SUPPRESS)
    parser.add_argument('--model',  type=str, help=argparse.SUPPRESS)
    parser.add_argument('--stats',  type=str, help=argparse.SUPPRESS)
    parser.add_argument('--ref_wt', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--ref_mut', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--targ_wt', type=str, help=argparse.SUPPRESS)

    # Optional args
    parser.add_argument('--output', type=str, default=input_params["output_dir"])
    parser.add_argument(
		'--suppress', type=str, action='append', default=input_params["suppressive_path"]
	)

    args = parser.parse_args()

    # Help first
    if args.help:
        print_custom_help()

    # Require --mode --action
    if args.mode is None or args.action is None:
        print("error: the arguments --mode and --action are required", file=sys.stderr)
        print_custom_help()

    # Validate --mode and --action
    if args.mode == "norm":
        if args.action == 'train':
            if not args.genome or not args.input:
                print("error: --genome and --input are required in normal train", file=sys.stderr)
                print_custom_help()

        elif args.action == 'prediction':
            if not args.model or not args.stats or not args.genome:
                print("error: --model, --stats, and --genome are required in normal prediction", file=sys.stderr)
                print_custom_help()

    elif args.mode == "cross_spe":
        if args.action == 'train':
            if not args.genome or not args.ref_wt or not args.ref_mut:
                print("error: --genome, --ref_wt and --ref_mut are required in cross-species train", file=sys.stderr)
                print_custom_help()

        elif args.action == 'prediction':
            if not args.model or not args.stats or not args.genome or not args.targ_wt:
                print("error: --model, --stats, --genome and --targ_wt are required in cross-species prediction",
                      file=sys.stderr)
                print_custom_help()

    return args, config


def run_deepPeak():
    cmd_args, config = parse_arguments()

    data_params = config["data_params"]
    train_params = config["train_params"]
    mode = cmd_args.mode
    action = cmd_args.action
    output_dir = cmd_args.output
    suppressive_paths = cmd_args.suppress

    print("\n" + "=" * 70)
    print("DBAT deepPeak - Sequencing Peak Predictor")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Action: {action}")

    if mode == "norm":
        if action == 'train':
            genome_path = Path(cmd_args.genome).resolve()
            train_path = Path(cmd_args.input).resolve()
            os.makedirs(output_dir, exist_ok=True)

            print(f"1. Data parameters:")
            print(f"   - seq len: {data_params['seq_length']} bp")
            print(f"   - bin size: {data_params['bin_size']} bp")
            print(f"2. Train parameters:")
            print(f"   - batch size: {train_params['batch_size']}")
            print(f"   - epochs: {train_params['epochs']}")
            print(f"   - Mixed Precision: {'Yes' if train_params['use_amp'] else 'No'}")
            print(f"3. Input & output:")
            print(f"   - genome: {genome_path}")
            print(f"   - train data: {train_path}")
            print(f"   - output dir: {output_dir}")
            print(f"   - suppressive data: {len(suppressive_paths)} file(s)")

            if suppressive_paths:
                for i, path in enumerate(suppressive_paths, 1):
                    print(f"     {i}. {path}")
            print("=" * 70)

            run_params = {
                'genome_path': str(genome_path),
                'train_path':  str(train_path),
                'output_dir':  output_dir,
                'suppressive_paths': suppressive_paths,
                'data_params': data_params,
                'train_params': train_params
            }
            norm_train(**run_params)

        elif action == 'prediction':
            model_path = Path(cmd_args.model).resolve()
            stats_path = Path(cmd_args.stats).resolve()
            genome_path = Path(cmd_args.genome).resolve()
            os.makedirs(output_dir, exist_ok=True)

            print(f"1. Prediction parameters:")
            print(f"   - model: {model_path}")
            print(f"   - stats: {stats_path}")
            print(f"   - genome: {genome_path}")
            print(f"   - output dir: {output_dir}")
            print("=" * 70)

            run_params = {
                'genome_path': str(genome_path),
                'model_path':  str(model_path),
                'stats_path':  str(stats_path),
                'output_dir':  output_dir,
                'data_params': data_params,
                'train_params': train_params
            }
            norm_prediction(**run_params)

    elif mode == "cross_spe":
        if action == 'train':
            genome_path = Path(cmd_args.genome).resolve()
            ref_wt_path = Path(cmd_args.ref_wt).resolve()
            ref_mut_path = Path(cmd_args.ref_mut).resolve()

            os.makedirs(output_dir, exist_ok=True)

            print(f"1. Data parameters:")
            print(f"   - seq len: {data_params['seq_length']} bp")
            print(f"   - bin size: {data_params['bin_size']} bp")
            print(f"2. Train parameters:")
            print(f"   - batch size: {train_params['batch_size']}")
            print(f"   - epochs: {train_params['epochs']}")
            print(f"   - Mixed Precision: {'Yes' if train_params['use_amp'] else 'No'}")
            print(f"3. Input & output:")
            print(f"   - genome: {genome_path}")
            print(f"   - ref species wt: {ref_wt_path}")
            print(f"   - ref species mut: {ref_mut_path}")
            print(f"   - output dir: {output_dir}")
            print("=" * 70)

            run_params = {
                'genome_path': str(genome_path),
                'ref_wt_path': str(ref_wt_path),
                'ref_mut_path': str(ref_mut_path),
                'output_dir': output_dir,
                'data_params': data_params,
                'train_params': train_params
            }
            cross_train(**run_params)


        elif action == 'prediction':
            model_path = Path(cmd_args.model).resolve()
            stats_path = Path(cmd_args.stats).resolve()
            genome_path = Path(cmd_args.genome).resolve()
            targ_wt_path = Path(cmd_args.targ_wt).resolve()

            os.makedirs(output_dir, exist_ok=True)

            print(f"1. Prediction parameters:")
            print(f"   - model: {model_path}")
            print(f"   - stats: {stats_path}")
            print(f"   - genome: {genome_path}")
            print(f"   - targ wt: {targ_wt_path}")
            print(f"   - output dir: {output_dir}")
            print("=" * 70)

            run_params = {
                'genome_path': str(genome_path),
                'targ_wt_path': str(targ_wt_path),
                'model_path': str(model_path),
                'stats_path': str(stats_path),
                'output_dir': output_dir,
                'data_params': data_params,
                'train_params': train_params
            }
            cross_prediction(**run_params)



if __name__ == "__main__":
    run_deepPeak()