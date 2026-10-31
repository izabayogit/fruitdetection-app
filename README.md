# FruitVision — Streamlit Fruit Detection

## Files
- `app.py` — Streamlit application
- `fruitdetection.pt` — your trained YOLO model
- `requirements.txt` — Python dependencies

## 1. Install Python
Use Python 3.10 or 3.11 if possible.

## 2. Open the project folder in PowerShell
Change directory to the folder containing these files, for example:

```powershell
cd "C:\path\to\fruit_detection_streamlit"
```

## 3. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open it in your browser.

## Important
Keep `fruitdetection.pt` in the same folder as `app.py`. The app reads the class names from the model automatically, so you do not need to manually enter fruit class names.

This app expects an Ultralytics-compatible YOLO `.pt` detection model. If your model was trained with a different framework, its inference code will need to be adapted.
