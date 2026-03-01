"""Inference module `src/inference/run_inference.py` that converts inputs into runtime prediction outputs."""

# Import `argparse` to support computations in this stage of output generation.
import argparse
# Import `json` to support computations in this stage of output generation.
import json

# Import symbols from `src.inference.predictor` used in this stage's output computation path.
from src.inference.predictor import ASDPredictor


# Define a reusable pipeline function whose outputs feed later steps.
def main():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `parser` for subsequent steps so the returned prediction payload is correct.
    parser = argparse.ArgumentParser(description="Run ASD inference on a video")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--video", type=str, required=True)
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--checkpoint", type=str, default="results/asd_pipeline_model.pth")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--config", type=str, default=None)
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--device", type=str, default=None)
    # Set `args` for subsequent steps so the returned prediction payload is correct.
    args = parser.parse_args()

    # Set `predictor` to predicted labels/scores that are reported downstream.
    predictor = ASDPredictor(args.checkpoint, config_path=args.config, device=args.device)
    # Set `result` for subsequent steps so the returned prediction payload is correct.
    result = predictor.predict_video(args.video)
    # Log runtime values to verify that output computation is behaving as expected.
    print(json.dumps(result, indent=2))


# Branch on `__name__ == "__main__"` to choose the correct output computation path.
if __name__ == "__main__":
    # Call `main` and use its result in later steps so the returned prediction payload is correct.
    main()
