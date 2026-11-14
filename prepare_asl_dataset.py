import zipfile
import random
import shutil
from pathlib import Path

ZIP = Path.home() / "Downloads" / "archive.zip"
OUT = Path("datasets/asl_alphabet")

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
random.seed(42)

if not ZIP.exists():
    raise FileNotFoundError(ZIP)

with zipfile.ZipFile(ZIP) as z:
    grouped = {c: [] for c in CLASSES}
    official = []

    for name in z.namelist():
        parts = name.replace("\\", "/").split("/")
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if len(parts) >= 2 and parts[-2] in grouped:
            grouped[parts[-2]].append(name)

        elif len(parts) >= 2 and parts[-1].upper().endswith("_TEST.JPG"):
            letter = parts[-1][0].upper()
            if letter in grouped:
                official.append((letter, name))

    for letter in CLASSES:
        files = grouped[letter]
        if len(files) != 3000:
            raise RuntimeError(
                f"{letter}: expected 3000 images, found {len(files)}"
            )

        random.shuffle(files)

        splits = {
            "train": files[:2400],
            "validation": files[2400:2700],
            "test": files[2700:]
        }

        for split, entries in splits.items():
            folder = OUT / split / letter
            folder.mkdir(parents=True, exist_ok=True)

            for i, entry in enumerate(entries):
                suffix = Path(entry).suffix.lower()
                target = folder / f"{letter}_{i:04d}{suffix}"

                with z.open(entry) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)

        print(f"{letter}: 2400 train, 300 validation, 300 test")

    for letter, entry in official:
        folder = OUT / "official_test" / letter
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(entry).name

        with z.open(entry) as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)

print("\nDataset preparation completed.")
print("Output:", OUT.resolve())
