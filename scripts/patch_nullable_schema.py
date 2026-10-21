"""Let an existing skill's store accept null where its spec does.

Applies ``widen_nullable_fields`` to ``<skill>/schema.json`` using
``<skill>/api.json``, rewrites schema.json in place, and prints each property
it widened. Idempotent: a second run widens nothing.

Usage:
    uv run python scripts/patch_nullable_schema.py <skill-dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulation_harness.skills.generation.stages.schema import (
    nullable_widenings,
    widen_nullable_fields,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skill_dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the widenings, write nothing"
    )
    args = parser.parse_args(argv)

    schema_path = args.skill_dir / "schema.json"
    spec_path = args.skill_dir / "api.json"
    for path in (schema_path, spec_path):
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 2
    schema = json.loads(schema_path.read_text())
    spec = json.loads(spec_path.read_text())

    widened = nullable_widenings(schema, spec)
    for def_name, prop in widened:
        print(f"{def_name}.{prop}")
    if not widened:
        print("nothing to widen")
        return 0
    if not args.dry_run:
        widen_nullable_fields(schema, spec)
        schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
