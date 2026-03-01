# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import argparse
import json

from src.inference.predictor import ASDPredictor


def main():
    parser = argparse.ArgumentParser(description="Run ASD inference on a video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="results/asd_pipeline_model.pth")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor = ASDPredictor(args.checkpoint, config_path=args.config, device=args.device)
    result = predictor.predict_video(args.video)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

