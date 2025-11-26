Salient Object Detection (SOD) – From Scratch

This project implements a complete Salient Object Detection (SOD) system using a custom convolutional neural network built entirely from scratch. The model identifies the main salient object in an image and outputs a binary mask. The entire machine learning pipeline was implemented manually, including dataset loading, preprocessing, augmentation, model architecture, training, evaluation, and a demo application.

Project Structure
SOD_Project/
│
├── src/
│ ├── data_loader.py  
│ ├── model.py  
│ ├── losses.py  
│ ├── train.py  
│ ├── evaluation.py  
│
├── data/
│ └── DUTS/  
│
├── final_model.pth  
├── demo.py  
├── README.md

1. Requirements

The project is implemented using Python and PyTorch. Install required packages:

pip install torch torchvision torchaudio
pip install numpy pillow matplotlib scikit-learn tqdm opencv-python
pip install gradio

Make sure to use a virtual environment, especially when running on macOS.

2. Dataset

This project uses the DUTS dataset, a standard dataset for salient object detection.

~10,553 training images (DUTS-TR)

~5,019 testing images (DUTS-TE)

Each image has a ground-truth saliency mask.
All images and masks are resized to 128×128 for training efficiency.

3. Model Architecture

A custom encoder–decoder CNN was implemented from scratch, meeting the project requirements:

Input: RGB image, size 128×128

Encoder: 3–4 convolutional layers with ReLU and MaxPooling

Decoder: 3–4 ConvTranspose2D layers with ReLU

Output: 1-channel sigmoid mask

No pretrained weights

The model is intentionally lightweight to train efficiently on CPU.

4. Training Configuration

The training loop includes:

Combined loss: Binary Cross-Entropy + 0.5 × (1 − IoU)

Optimizer: Adam, learning rate = 5e-4

Batch size: 8

Early stopping with patience = 5

Up to 30 epochs

Run training:

python -m src.train

This trains the model and saves:

final_model.pth

5. Evaluation

Evaluation metrics include:

Intersection-over-Union (IoU)

Precision

Recall

F1-score

Dice coefficient

Visualizations of predictions (input, mask, overlay)

Run evaluation:

python -m src.evaluation

Results and output images are saved under:

outputs/visuals/

6. Demo Application

A Gradio demo application is provided for easy interaction.

Run:

python demo.py

Features:

Upload any image

Predict salient object mask

View input, predicted mask, and overlay

See inference time per image

This demo is used for the final presentation.

7. Summary

This project demonstrates an end-to-end implementation of a Salient Object Detection system without relying on pretrained models. It includes:

Custom dataloaders and preprocessing

A from-scratch CNN model

Training and evaluation pipeline

Metric computation

Visualization and demo interface

The improved model achieved strong results on the DUTS test set, showing good generalization and clear detection of salient objects.
