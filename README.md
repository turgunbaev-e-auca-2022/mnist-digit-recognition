# MNIST Digit Recognition

**Emirlan Turgunbaev — academic project, adapted for a standalone Python workflow with AI assistance.**

This project compares a single softmax classification layer with a convolutional neural network for recognizing handwritten digits (0–9). The architectures come from my original coursework notebook; the standalone script adds portable paths, a validation split, fixed seeds, model saving, and a JSON results file.

## Data and method

MNIST downloads automatically through TensorFlow/Keras on the first run (internet required). Images are 28 × 28 grayscale pixels, normalized to [0, 1]. The script uses 50,000 training, 10,000 validation, and 10,000 held-out test images. Both models use RMSprop and sparse categorical cross-entropy. The CNN has three convolutional layers, two pooling layers, and a dense classification head. This is a standard MNIST exercise, not a novel architecture.

## Run

Use Python 3.11 in a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

A shorter trial:

```bash
python train.py --epochs 1
```

The script saves `outputs/baseline.keras`, `outputs/cnn.keras`, and `outputs/metrics.json`. The seed helps repeatability but does not guarantee identical results on every device.

## Results and verification

The original notebook contains saved test accuracies of **92.68%** for the softmax baseline and **99.14%** for the CNN. These are historical notebook outputs, not independently reproduced results. The new script uses a different training/validation split, so its results must be reported separately after a fresh run. No new accuracy is claimed for this version.

Preparation checks: Python syntax checked. Full training has not been run in the preparation environment. Installation and end-to-end execution still need verification on a machine with TensorFlow and dataset download access.

## Original coursework

`original/MId_Emir.ipynb` preserves the submitted notebook's code and text without stored outputs or local machine paths. The assignment includes an additional pretrained image-classification exercise; that part is not included in this standalone digit-recognition script. The notebook's course/template attribution is retained. Do not use its unrelated image-classification section as a claim of training EfficientNet from scratch.

## Limitations

MNIST performance does not demonstrate accuracy on arbitrary photos of handwriting. This repository does not include a custom dataset, a deployed service, or production benchmarking.

## References

- MNIST: https://yann.lecun.org/exdb/mnist/
- TensorFlow/Keras MNIST loader: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
