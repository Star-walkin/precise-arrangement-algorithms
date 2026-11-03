"""Write one immutable local freeze receipt before evaluation predictions."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = Path(r"C:\precise_arrangement_results\geometry_direction_20261103_v1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = ["PROTOCOL.md", "config.json", "inputs.json", "evaluation_inputs.json",
             "geometry_core.py", "run_geometry.py", "evaluation_adapter.py", "audit_outputs.py"]
    for name in files:
        if not (HERE / name).is_file():
            raise FileNotFoundError(name)
    pilot = OUTPUT / "pilot_003" / "run_receipt.json"
    if not pilot.is_file():
        raise FileNotFoundError(pilot)
    p = json.loads(pilot.read_text(encoding="utf-8"))
    if p["exit_code"] != 0 or p["completed_rows"] != 7:
        raise ValueError("Pilot 003 incomplete")
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_historical_evaluation_predictions",
        "method_id": "IR-GEOM-G1.0.0",
        "files": {name: sha(HERE / name) for name in files},
        "pilot_003_receipt_sha256": sha(pilot),
        "python": sys.version,
        "platform": platform.platform(),
        "output_root": str(OUTPUT),
        "network_heat_or_model_used": False,
        "target_reference_read_before_freeze": False,
    }
    dest = HERE / "freeze_receipt.json"
    with dest.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(dest)


if __name__ == "__main__":
    main()
