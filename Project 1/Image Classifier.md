
# Cat vs Dog Classifier using TensorFlow

This project demonstrates how to clean a dataset, load images, normalize data, build a Convolutional Neural Network (CNN), train the model, and visualize predictions using TensorFlow.

---

##  Step 1: Import Required Libraries

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import warnings
import logging
from PIL import ImageFile
```
tensorflow: Main ML library for building the CNN.

matplotlib.pyplot: Used to visualize predictions.

os, shutil: Help with directory and file handling.

warnings, logging: Suppress unnecessary logs.

PIL.ImageFile: Allows us to load and validate images
