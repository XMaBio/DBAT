#!/usr/bin/env python3
import sys
import argparse


def dispatch_deepPeak():
    try:
        from deepPeak.deepPeak import run_deepPeak

        # original argv: ['dbat', 'deepPeak', '-h']
        original_argv = sys.argv.copy()

        try:
            # Modified argv:  ['dbat', '-h']
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            run_deepPeak()

        finally:
            # Restore original argv
            sys.argv = original_argv

    except ImportError as e:
        print(f"[dbat] Error importing deepPeak module: {e}")
        sys.exit(1)



def dispatch_deepLoci():
    try:
        from deepLoci.deepLoci import run_deepLoci

        # original argv: ['dbat', 'deepLoci', '-h']
        original_argv = sys.argv.copy()

        try:
            # Modified argv:  ['dbat', '-h']
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            run_deepLoci()

        finally:
            # Restore original argv
            sys.argv = original_argv

    except ImportError as e:
        print(f"[dbat] Error importing deepLoci module: {e}")
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        help=""
    )

    subparsers.add_parser(
        "deepPeak",
        help="Predict sequencing peaks from DNA sequence",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers.add_parser(
        "deepLoci",
        help="Predict featured genomic loci",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ignore -h/--help
    args, unknown = parser.parse_known_args()

    # Show main help if no command provided
    if args.command is None:
        print("Usage: dbat <command> [options]\n")
        print("Commands:")
        print("  deepPeak    predict sequencing peaks.")
        print("  deepLoci    predict featured genomic loci.")
        print("\nUse 'dbat <command> --help' for more information on a specific command.")
        sys.exit(0)

    # Command dispatch
    if args.command == "deepPeak":
        dispatch_deepPeak()

    elif args.command == "deepLoci":
        dispatch_deepLoci()

    else:
        print(f"[dbat] Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()