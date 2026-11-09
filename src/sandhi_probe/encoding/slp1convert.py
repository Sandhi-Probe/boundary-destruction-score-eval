"""Convert the DCS and UoH raw samples to SLP1.

With no arguments, both raw files are converted into ``data/slp1/``:

* DCS remains TSV, with its Sanskrit ``text`` and ``lemmata`` columns converted.
* UoH is extracted from HTML as a plain-text SLP1 corpus; header metadata and
  editorial verse markers are omitted.

The raw files are never modified.

Examples:
    python slp1convert.py
    python slp1convert.py input.txt output.txt --plain
    python slp1convert.py data/raw/pada-and-analysis.dat dcs-slp1.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bs4 import BeautifulSoup
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/raw/pada-and-analysis.dat"
UOH_INPUT = ROOT / "data/raw/vinodini_p.html"
OUTPUT_DIR = ROOT / "data/slp1"
DEFAULT_OUTPUT = OUTPUT_DIR / "pada-and-analysis-slp1.dat"
UOH_OUTPUT = OUTPUT_DIR / "vinodini_p-slp1.txt"


def convert_text(text: str, source: str = sanscript.IAST) -> str:
    """Convert one string to SLP1 while preserving spaces and punctuation."""
    return transliterate(text, source, sanscript.SLP1)


def convert_plain(input_path: Path, output_path: Path) -> int:
    """Convert one IAST line at a time, retaining line endings and blanks."""
    converted = 0
    with input_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        for line in source:
            content, ending = line.rstrip("\r\n"), line[len(line.rstrip("\r\n")) :]
            destination.write(convert_text(content) + ending)
            converted += 1
    return converted


def convert_dcs(input_path: Path, output_path: Path) -> int:
    """Convert DCS Sanskrit columns and preserve numeric/structural columns."""
    converted = 0
    with input_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for row in reader:
            if len(row) > 4 and not (row and row[0] == "book"):
                row[4] = convert_text(row[4])  # text
                if len(row) > 5:
                    row[5] = convert_text(row[5])  # lemmata
                converted += 1
            writer.writerow(row)
    return converted


def convert_uoh(input_path: Path, output_path: Path) -> int:
    """Extract UoH Devanagari and convert corpus lines to SLP1."""
    with input_path.open("r", encoding="utf-8") as source:
        soup = BeautifulSoup(source, "html.parser")

    converted = 0
    metadata_started = False
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for line in soup.get_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if not metadata_started and stripped.startswith((
                "Book Name", "Author", "Project Name", "Center", "Typed by",
                "Proofcheck by", "Sandsplitted by", "Sandhi Matched by",
            )):
                continue
            metadata_started = True
            if not any("\u0900" <= char <= "\u097F" for char in line):
                continue
            # Editorial bars and numbered bars are not Sanskrit input.
            cleaned = line
            while "|" in cleaned:
                cleaned = cleaned.replace("|", "")
            destination.write(convert_text(cleaned, sanscript.DEVANAGARI) + "\n")
            converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="treat input as plain IAST text instead of DCS TSV",
    )
    args = parser.parse_args()

    if args.input is None and args.output is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dcs_count = convert_dcs(DEFAULT_INPUT, DEFAULT_OUTPUT)
        uoh_count = convert_uoh(UOH_INPUT, UOH_OUTPUT)
        print(f"Wrote {dcs_count} DCS rows to {DEFAULT_OUTPUT}")
        print(f"Wrote {uoh_count} UoH lines to {UOH_OUTPUT}")
        return
    if args.input is None or args.output is None:
        parser.error("provide both input and output, or provide neither to convert both raw files")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_plain(args.input, args.output) if args.plain else convert_dcs(args.input, args.output)
    print(f"Wrote {converted} converted lines to {args.output}")


if __name__ == "__main__":
    main()
