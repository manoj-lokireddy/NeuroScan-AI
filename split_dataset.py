import os
import shutil
import random

random.seed(42)

OUTPUT = "dataset"
CLASSES = ["yes", "no"]

# Create folders
for split in ["train", "validation", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT, split, cls), exist_ok=True)

# Split each class separately
for cls in CLASSES:

    source_folder = cls

    images = [
        f for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * 0.70)
    validation_count = int(total * 0.15)

    train_images = images[:train_count]
    validation_images = images[train_count:train_count + validation_count]
    test_images = images[train_count + validation_count:]

    for filename in train_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(OUTPUT, "train", cls, filename)
        )

    for filename in validation_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(OUTPUT, "validation", cls, filename)
        )

    for filename in test_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(OUTPUT, "test", cls, filename)
        )

    print(f"{cls}:")
    print(f"  Total      : {total}")
    print(f"  Train      : {len(train_images)}")
    print(f"  Validation : {len(validation_images)}")
    print(f"  Test       : {len(test_images)}")
    print()

print("Dataset split completed successfully!")