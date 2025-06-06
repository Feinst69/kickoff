# Handwritten Digit Classification Project

This repository contains a simple example of classifying handwritten digits using the MNIST dataset.

## Structure

- `theoretical_questions.md` – placeholder for theoretical answers extracted from the PDF.
- `notebooks/01_exploration.ipynb` – notebook for exploring the data.
- `notebooks/02_modeling.ipynb` – notebook for training a model.
- `train.py` – script to train a convolutional neural network and save it as `model.h5`.
- `predict.py` – script that loads the trained model and predicts the digit of an image provided on the command line.
- `app.py` – a small Flask web interface to upload an image and see the predicted digit.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Predict a digit from an image:
   ```bash
   python predict.py path/to/image.png
   ```
4. Launch the web application:
   ```bash
   python app.py
   ```

## Results

Running `train.py` will train a small CNN for five epochs on MNIST. Accuracy will depend on the runtime environment.

## Conclusion

This project demonstrates a minimal pipeline for loading data, training a model, making predictions via script and web interface, and presenting results.
