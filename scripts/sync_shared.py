"""Copy the shared client into each standalone plugin."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "shared/gmi_media.py").read_bytes()
    different = []
    for name in ("gmi-image", "gmi-video", "gmi-speech"):
        target = ROOT / name / "gmi_media.py"
        if args.check:
            if not target.exists() or target.read_bytes() != source:
                different.append(name)
        else:
            target.write_bytes(source)
    if different:
        print("Shared client differs: " + ", ".join(different))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
