import argparse
import json
from pathlib import Path

from video_pipeline.preprocess import load_config, prepare_video_dataset


def main():
    parser = argparse.ArgumentParser(description="Audit and sample labeled videos into source-grouped image frames.")
    parser.add_argument("--config", type=Path, default=Path("configs/keras_training.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-audit", action="store_true", help="Re-read and hash every source video.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    summary = prepare_video_dataset(config, config_path, force_audit=args.force_audit, workers=args.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
