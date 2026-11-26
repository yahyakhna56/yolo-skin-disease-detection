# Skin Disease Image Classification with YOLOv8 and Streamlit

A four-class **image-classification** demonstration built with a trained Ultralytics YOLOv8 classification checkpoint. The application accepts one image and displays the model's predicted class and class-score output.

> **Medical-use limitation:** This is an educational/research demonstration, not a medical diagnostic device. A model prediction cannot confirm or rule out a disease and must not be used to make treatment decisions or delay professional medical care.

## Classes

- Chickenpox
- Measles
- Monkeypox
- Normal

## Project structure

```text
Skin_Disease_Classification_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   └── best.pt                 # Add your trained checkpoint here
└── notebooks/
    └── Skin_Disease_Classification_YOLOv8.ipynb
```

## 1. Add your trained model

The trained checkpoint is not included in this ZIP because `best.pt` was not attached to the conversation when this package was generated.

Copy your own trained `best.pt` file into the `models` folder so the final path is:

```text
models/best.pt
```

The application also supports `best.pt` beside `app.py`, but `models/best.pt` is the recommended location. Use the checkpoint produced by your classification training notebook (`yolov8n-cls.pt` workflow), not an object-detection checkpoint.

## 2. Run locally on macOS, Windows, or Linux

Use a supported Python version compatible with the selected Ultralytics and PyTorch releases. Streamlit Community Cloud currently defaults to Python 3.12; select the same Python version for local testing and deployment where possible.

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\\Scripts\\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application from the project root:

```bash
streamlit run app.py
```

## 3. Upload to GitHub

1. Extract this ZIP file.
2. Copy `best.pt` into `models/best.pt`.
3. Create a new GitHub repository.
4. Upload the **contents of the extracted project folder** (not just the ZIP file).
5. Confirm that `app.py`, `requirements.txt`, and `models/best.pt` are in the repository.

If GitHub rejects `best.pt` because it exceeds GitHub's regular-file size limit, use Git Large File Storage (Git LFS) for the model file. Do not commit your dataset ZIP or the full image dataset unless you specifically intend to publish it.

## 4. Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose **Create app** and select your GitHub repository and branch.
3. Set the app entrypoint to `app.py` (project root).
4. In **Advanced settings**, select Python 3.12 or another Python version supported by the packages and used for your local test.
5. Deploy.

Streamlit installs packages from `requirements.txt`. The model file must be present in the repository at `models/best.pt` (or at the project root as `best.pt`) because the deployed app cannot read files from your Colab runtime or your Mac.

## 5. Important notes

- The notebook in `notebooks/` is for training and evaluation in Google Colab. Streamlit runs `app.py`; it does not run the notebook to make predictions.
- The app loads the existing `best.pt` checkpoint and does not retrain the model.
- This project is an image classifier, not an object detector. It does not produce bounding boxes.
- A high model confidence score does not establish medical certainty.
