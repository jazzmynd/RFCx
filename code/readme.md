# Basic Instructions
1. Run `get-class-frequencies.ipynb` to generate a data file containing the min/max frequency indices for each class. This is needed for data augmentation defined in `datagen.py`.
2. Spectrogram data can be generated using the function `wave_to_mel_spec()` in `specinput.py`.
3. Raw or augmented spectrograms can be visualized with `check-train-generator.ipynb`.
4. A CNN can be trained for spectrogram labeling with `train.ipynb`.

# Files
1. `get-class-frequncies.ipynb`
This notebook generates a `.npy` file containing the indices of the min and max frequencies of each class in a mel-spectrogram based on the parameters in `specinput.py` and the metadata in `class-meta.csv`. This file is required for the CutMix augmentation applied in `datagen.py`, so generate this file first. 

2. `check-train-generator.ipynb`
This notebook sets up a data generator, loads N batches into memory, and visualizes examples of a specific class.

3. `train.ipynb`
Sets up train+validation spectrogram image generators, defines a CNN model and training hyperparameters, and fits the model

4. `specinput.py`
This file stores the parameters and code for reading audio and computing spectrograms.

5. `datagen.py`
Defines a class for generating batches of spectrogram images and on-the-fly augmentation

6. `learningrate.py`
Contains a function to define a learning rate curve with warmup and cosine decay, and a Keras Callback class for applying the schedule during fitting

