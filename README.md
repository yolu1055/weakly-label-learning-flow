# Wealy Label Learning Flow

This is a Python implementation of the Weakly Supervised Label Learning Flow introducted in the paper 

"[Weakly Supervised Label Learning Flows](https://arxiv.org/abs/2302.09649)". You Lu, Chidubem Arachie and Bert Huang. 2022.

An example dataset, i.e., [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), is at `./datasets`, which can be used to run the code directly.

The code for processing Fashion-MNIST data is adapted from the [ALL method](https://github.com/VTCSML/Adversarial-Label-Learning). For running experiments on text datasets, use the same way as [CLL](https://github.com/VTCSML/Constrained-Labeling-for-Weakly-Supervised-Learning/blob/main/generate_weak_signals.ipynb) to process the raw text datasets.
The code for developing flow models is adapted from [dpf-nets](https://github.com/Regenerator/dpf-nets).

## Requirements:

This code was tested using the the following libraries.

- Python 3.7
- Pytorch 1.7.0

## Running

- In `main.py`, specify the dataset path, i.e., `--data_path` and the output dir, i.e., `--out_root`.
- Run `main.py`.

## Contact
Feel free to send me an email, if you have any questions or comments.

