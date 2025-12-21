import os
import pathlib
import joblib
import subprocess
from typing import Tuple

ROOT = pathlib.Path(__file__).parent
MODEL_DIR = ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"


def ensure_pipeline() -> object:
    """Load the saved pipeline. If it's missing, run train.py to create it, then load."""
    if not PIPELINE_PATH.exists():
        print("Pipeline not found. Running training to create model...")
        # Run training script
        subprocess.run(["python", str(ROOT / "train.py")], check=False)

    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        return pipeline
    else:
        raise FileNotFoundError(f"Could not find or create pipeline at {PIPELINE_PATH}")


def predict(text: str) -> Tuple[str, float]:
    """Return (label, probability) for the given text headline."""
    pipeline = ensure_pipeline()
    label = pipeline.predict([text])[0]
    probs = pipeline.predict_proba([text])[0]
    # find index of predicted label
    idx = list(pipeline.classes_).index(label)
    prob = float(probs[idx])
    return label, prob


if __name__ == "__main__":
    # quick CLI test
    sample = "You won't believe what this politician said at the rally"
    lbl, p = predict(sample)
    print(sample)
    print(lbl, p)
