import argparse
import sys

# 1.)Importing functions—I'm calling the logic I wrote in my other files.
try:
    from train import run_training
    from eval import run_eval
except ImportError as e:
    print(f"   [ERROR] Missing module: {e}. Ensure all files are in the same folder.")
    sys.exit(1)

# 2.)CLI setup—This lets me control my RTX 4050 experiments from the terminal.
# Example: python main.py --mode train --batch 128 --lr 5e-4.
def get_args():
    parser = argparse.ArgumentParser(description="LogicSynth Transformer Control Center")
    parser.add_argument("--mode",   type=str,   default="train", choices=["train", "eval"])
    parser.add_argument("--batch",  type=int,   default=None, help="Override default batch size.")
    parser.add_argument("--lr",     type=float, default=None, help="Override default learning rate.")
    parser.add_argument("--epochs", type=int,   default=None, help="Override default max epochs.")
    return parser.parse_args()

# 3.)Main execution—Routing the logic based on the --mode flag.
if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        print("\n   Launching LogicSynth Training Session...")
        # Passing the overrides from CLI directly to the train function.
        run_training(batch=args.batch, lr=args.lr, epochs=args.epochs)
    
    else:
        print("\n   Launching LogicSynth Full Evaluation Suite...")
        # Since eval.py is now a function, we just call it.
        # It handles its own model init and checkpoint loading.
        run_eval()

    print("\n   Process Complete.\n")