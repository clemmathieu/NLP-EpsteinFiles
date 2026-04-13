"""
main.py - Epstein Investigation Pipeline

    Stage 1 - Binary classification - data/binary_classified.csv
    Stage 2 - Offense classification - data/classified_emails.csv
                                         data/embeddings.npy

Usage

    python main.py # DEMO mode (first 200 threads, ~5 min CPU)
    python main.py --full # Full run   (~60 min CPU / ~15 min GPU)

After the pipeline completes, launch the dashboard with:
    streamlit run app.py
"""

import argparse
import sys
import os
import time

# Ensure the project root is on the path when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import binary_classification, offense_classification


#  CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epstein Investigation - NLP Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py # DEMO mode (200 threads)\n"
            "  python main.py --full # Full dataset\n"
            "  python main.py --skip-stage1  # Re-run Stage 2 only"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on the full dataset (default: DEMO mode, first 200 threads)",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 and load existing binary_classified.csv",
    )
    return parser.parse_args()

#  Helpers

def _header(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _section(text: str) -> None:
    print(f"\n[•] {text}")
    print("-" * 50)

#  Main 

def main() -> None:
    args    = parse_args()
    demo    = not args.full
    t_start = time.time()

    _header("EPSTEIN INVESTIGATION PIPELINE")

    if demo:
        print("\n⚠️  Running in DEMO mode (first 200 threads).")
        print("   Pass --full to classify the entire dataset.")

    #  Stage 1 
    if args.skip_stage1:
        stage1_path = binary_classification.OUTPUT_PATH
        if not os.path.exists(stage1_path):
            print(f"\n❌  Cannot skip Stage 1: {stage1_path} not found.")
            sys.exit(1)
        print(f"\n⏭️  Skipping Stage 1 - loading {stage1_path}")
    else:
        _section("Stage 1 - Binary Classification")
        binary_classification.run(demo_mode=demo)

    # Stage 2 
    _section("Stage 2 - Offense Classification + NER + Embeddings")
    offense_classification.run()

    elapsed = time.time() - t_start
    _header(f"✅  PIPELINE COMPLETE  ({elapsed / 60:.1f} min)")

    print(
        "\nTo launch the investigation dashboard run:\n\n"
        "    streamlit run app.py\n"
    )

if __name__ == "__main__":
    main()
