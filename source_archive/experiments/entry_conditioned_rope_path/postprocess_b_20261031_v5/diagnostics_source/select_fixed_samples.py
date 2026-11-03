"""Freeze open development examples using metadata only; no image/prediction reads."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "postprocess_b_20261031_v4" / "inputs.json"
SALT = "H5_open_check_20261031_v1"
COUNTEREXAMPLE = "dual_20260923_101517__pair0195__right"

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    config = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    jobs = config["jobs"]
    development_ids = list(dict.fromkeys(config["pilot_ids"] + [COUNTEREXAMPLE]))
    assert len(jobs) == 292 and len(development_ids) == 14
    extra = []
    for camera in ("left", "right"):
        candidates = [j for j in jobs if j["camera_role"] == camera and j["id"] not in development_ids]
        groups = {j["source_group"] for j in candidates}
        # Multiple-image acquisition groups, not single ruler stills. This does
        # not turn open historical development material into an independent test.
        groups = [g for g in groups if sum(j["source_group"] == g for j in candidates) >= 2]
        groups.sort(key=lambda g: digest(SALT + "/group/" + g))
        for group in groups[:4]:
            choices = [j for j in candidates if j["source_group"] == group]
            choice = min(choices, key=lambda j: digest(SALT + "/image/" + j["id"]))
            extra.append({k: choice[k] for k in ("id", "source_group", "camera_role", "source_time_s", "image_path", "image_sha256", "dataset_use_status")})
    assert len(extra) == 8 and len({j["source_group"] for j in extra}) == 8
    result = dict(
        schema="metadata_only_fixed_development_and_postfreeze_check_v1",
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        allowed_images=292, allowed_source_groups=len({j["source_group"] for j in jobs}),
        development_ids=development_ids, extra_open_check=extra,
        selection_rule="Exclude14 development IDs; per camera, SHA256 salted group order selects4 multi-image groups, then salted minimum image-ID hash selects1 each.",
        salt=SALT, source_group_isolation_from_development=False,
        independent_blind_test=False, image_content_read_for_selection=False,
        predictions_read_for_selection=False, labels_read=False,
        use="Only inspect these8 after H5 parameters freeze; no subsequent tuning from their outcomes. Existing open development status remains unchanged.",
    )
    output = HERE / "fixed_samples_metadata.json"
    if output.exists():
        assert json.loads(output.read_text(encoding="utf-8")) == result
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"development_count":14,"extra_ids":[j["id"] for j in extra]},ensure_ascii=True))

if __name__ == "__main__":
    main()
