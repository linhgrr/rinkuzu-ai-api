from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the FastAPI OpenAPI schema to JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "openapi.json",
        help="Output path for the generated OpenAPI schema JSON.",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
