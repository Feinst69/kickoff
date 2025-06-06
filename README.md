# Handwritten Digit Classification Project

This repository demonstrates an end‑to‑end pipeline for classifying handwritten digits from the MNIST dataset. It includes the data exploration notebooks, training scripts and a small Flask application.

## Structure

- `theoretical_questions.md` – answers to theory questions about neural networks.
- `notebooks/01_exploration.ipynb` – notebook for exploring the data.
- `notebooks/02_modeling.ipynb` – notebook for training experiments.
- `train.py` – grid search script that trains both a CNN and a multi-layer perceptron (MLP) and saves the best model as `model.h5`. Training curves are exported as PNG images.
- `predict.py` – command line prediction script.
- `app.py` – Flask web interface to upload an image and see the predicted digit.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model with grid search:
   ```bash
   python train.py
   ```
3. Predict a digit from an image:
   ```bash
   python predict.py path/to/image.png
   ```
4. Launch the web application (now exposing a prediction API):
   ```bash
   python app.py
   ```

Running `train.py` performs a grid search over several hyper‑parameters,
saving the best performing model to `model.h5`. The script also prints a
confusion matrix and classification report for the final model and stores
PNG images of loss and accuracy for each configuration.
