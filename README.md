# Vehicle Image Classification using CNN (2 classes for now)

This project currently includes **real-world images** for:
- Car
- Motorbike

The Bicycle dataset has not been added yet. Once you provide `bicycle.tar.gz`, it can be added and the project can be switched to three classes.

## Dataset counts

| Class | Total images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Car | 638 | 446 | 95 | 97 |\n| Motorbike | 503 | 352 | 75 | 76 |\n
The images were split into mutually exclusive training, validation, and test sets (approximately 70% / 15% / 15%). Mac metadata files were excluded.

## Project structure

```text
Vehicle_CNN_Classification/
├── CNN_Vehicle_Classification.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── Dataset/
│   ├── train/Car/ and Motorbike/
│   ├── validation/Car/ and Motorbike/
│   └── test/Car/ and Motorbike/
└── models/
```

## Run on Windows

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

Or open `CNN_Vehicle_Classification.ipynb` in Jupyter Notebook and run cells from top to bottom.

## Predict an image

```bash
python predict.py "C:\path\to\vehicle.jpg"
```

## Important

This version trains a **two-class model** (Car and Motorbike) because those are the two datasets currently provided. Do not use it as a three-class model until Bicycle images are added and the class list is updated. Test accuracy must be measured by running the training script; it is not pre-determined.
