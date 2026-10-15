"""Check the public HepG2 processed release and recompute panel recovery."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

def verify(release):
    release = Path(release)
    for line in (release / "SHA256SUMS.txt").read_text("utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        actual = hashlib.sha256((release / name.strip()).read_bytes()).hexdigest()
        if digest != actual:
            raise ValueError(f"Checksum mismatch: {name}")
    aligned = pd.read_csv(release / "hepg2_aligned_threshold015.csv.gz", index_col=0)
    assert aligned.shape == (18, 9960) and aligned.index.is_unique
    assert np.isfinite(aligned.to_numpy()).all()
    assert sum(aligned.index == "MI-773") == 1
    predictions = pd.read_csv(release / "hepg2_vae_predictions_long.csv")
    assert len(predictions) == 36 and not predictions.duplicated(["Compound", "branch"]).any()
    summary = []
    for branch, dimensions, expected in [("average", 194, 14), ("supcon", 257, 15)]:
        embedding = pd.read_csv(release / f"hepg2_{branch}_projected_embeddings.csv", index_col=0)
        assert embedding.shape == (18, dimensions)
        assert embedding.index.tolist() == aligned.index.tolist()
        assert np.isfinite(embedding.to_numpy()).all()
        panel = predictions.loc[predictions.branch.eq(branch)]
        assert len(panel) == 18 and set(panel.Compound) == set(aligned.index)
        recovered = int(panel.predicted_class.eq("Antineoplastic").sum())
        assert recovered == expected
        assert np.array_equal(panel.predicted_class.eq("Antineoplastic"),
                              panel.prob_antineoplastic.gt(panel.decision_threshold))
        summary.append({"branch": branch, "profiles": 18, "dimensions": dimensions,
                        "recovered_positive": recovered, "recovery_fraction": recovered / 18})
    return {"checksums_match": True, "predictions": 36, "branches": summary}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = verify(args.release_dir)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps(result, indent=2))
