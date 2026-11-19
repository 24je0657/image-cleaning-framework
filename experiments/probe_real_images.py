import argparse
import shutil
from pathlib import Path

CLASSES = ["cat", "dog"]
SOURCE = Path("data/raw/Animal")
PROBE_ROOT = Path("/tmp/image_cleaning_probe")


def create_dataset(per_class):
    dataset_dir = PROBE_ROOT / f"dataset_{per_class}"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ["train", "val"]:
        for cls in CLASSES:
            source_dir = SOURCE / split / cls
            target_dir = dataset_dir / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)

            images = sorted(
                p for p in source_dir.iterdir()
                if p.is_file()
            )

            if len(images) < per_class:
                raise ValueError(
                    f"Not enough images in {source_dir}"
                )

            for image in images[:per_class]:
                shutil.copy2(image, target_dir / image.name)

    return dataset_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class",
        type=int,
        required=True,
        choices=[30, 60],
    )
    args = parser.parse_args()

    dataset_dir = create_dataset(args.per_class)

    print(f"Dataset created: {dataset_dir}")
    print(f"Classes: {CLASSES}")
    print(f"Images per class per split: {args.per_class}")


if __name__ == "__main__":
    main()
