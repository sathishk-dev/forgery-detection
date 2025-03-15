# Forgery Detection

A deep learning-based image forgery detection system using the DPMSN model, implemented with Flask for a web-based interface.

## Project Structure
```
Forgery-Detection/
│── dataset/                # Dataset Folder
│   ├── images/             # Original images
│   ├── masks/              # Ground truth forgery masks
│── models/                 # Saved models
│── static/                 # Flask UI assets
│── templates/              # HTML templates for Flask
│── app.py                  # Flask Web App
│── dpmsn_model.py          # DPMSN Model Definition
│── dataset_loader.py       # Data Preprocessing & Loading
│── train.py                # Model Training Script
│── evaluate.py             # Model Evaluation Script
│── requirements.txt        # Required Packages
│── README.md               # Project Documentation
```

## Dataset Setup

Download CASIA 2 dataset from:
[CASIA 2 Dataset - Kaggle](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset/data)

### CASIA v2 Dataset Structure
The dataset contains the following folders:

- `Au/` → Authentic (real) images
- `Tp/` → Tampered (forged) images
- `CASIA2 Groundtruth/` → Binary masks for tampered images

### Organizing the Dataset
To use the dataset correctly, organize the files as follows:

#### Step 1: Move Images
✅ Copy all images from:
- `Au/` → Move all images into `dataset/images/`
- `Tp/` → Move all images into `dataset/images/`

#### Step 2: Move Masks
✅ Copy all ground truth masks from `CASIA2 Groundtruth/` into `dataset/masks/`

## Installation

### Step 1: Install Dependencies

You can install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

Alternatively, install them manually:

```sh
pip install torch torchvision torchaudio
pip install numpy opencv-python matplotlib flask
```

### Step 2: Preprocess Dataset

Run the dataset preprocessing script:

```sh
python dataset_loader.py
```

## Training the Model

To train the DPMSN model, run:

```sh
python train.py
```

#### Expected Console Output (Training):
```
Epoch [1/10], Loss: 0.3543
Epoch [2/10], Loss: 0.3011
...
Training Completed!
```
After training completes, check the `models/` folder for the file `dpmsn_model.pth`.

## Evaluating the Model

To evaluate the model's performance, run:

```sh
python evaluate.py
```

#### Expected Console Output (Evaluation):
```
Average F1 Score: 0.91
```

## Running the Web Application

To launch the Flask web app, run:

```sh
python app.py
```

Then, open your browser and visit:

```
http://127.0.0.1:5000/
```

### Web App Features:
- Upload an image
- View forged regions highlighted

🚀 Enjoy detecting forgeries with deep learning! 🚀

