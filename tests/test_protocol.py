import json
from pathlib import Path


def test_registered_splits_are_disjoint():
    root = Path(__file__).resolve().parents[1] / "data" / "manifests"
    for split in (1, 2, 3):
        manifest = json.loads((root / f"split_{split}.json").read_text())
        known = set(manifest["known_classes"])
        rogue = set(manifest["rogue_classes"])
        assert not known & rogue
        assert len(known) == 9
        assert len(rogue) == 8
        assert manifest["status"] in {"development", "FROZEN_BEFORE_OPEN_SET_TEST"}
