# Fake News Detector (Streamlit demo)

This is a small demo project that classifies news headlines as REAL or FAKE.

Files added:
- `data/news_headlines.csv` — sample labeled headlines used for training (toy dataset for demo)
- `train.py` — trains a TF-IDF + Logistic Regression pipeline and saves it to `model/pipeline.pkl`
- `model.py` — helper to load (or auto-train) the pipeline and predict
- `app.py` — Streamlit app. Run with `streamlit run app.py`
- `requirements.txt` — dependencies

Quick setup (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes
- This demo uses a tiny toy dataset included in `data/` for quick local testing. For a production-grade model you should train on a much larger, curated dataset and add robust evaluation and monitoring.
- The first time you run the app it will auto-train a small model if `model/pipeline.pkl` does not exist.

Made by Tejvir and Vishesh
