# MicroNAS — Memory-Constrained Hyperparameter Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MojtabaMohasel/MicroNAS/blob/main/Examples/AReM.ipynb)

This repository provides a customized version of **KerasTuner**, an easy-to-use and scalable hyperparameter optimization framework for Keras/TensorFlow models — extended with **memory-aware model search** for deployment-constrained platforms (e.g., microcontrollers and embedded devices).

With this version, you can:

✔️ Define your search space using standard KerasTuner syntax  
✔️ Run Random Search, Bayesian Optimization, or Hyperband  
✔️ **Enforce model memory limits using `max_model_size`**  
✔️ **Prevent tuner crashes using `max_consecutive_failed_trials`**  
✔️ Track how many candidate models exceeded the constraint  

This makes it especially useful for workflows targeting **TinyML, ESP32-class devices, MCUs, and edge AI deployment**.

Official KerasTuner Website: https://keras.io/keras_tuner/

---



## Installation

This fork requires **Python 3.8+** and **TensorFlow 2.0+**.

Install directly from the repository:

```bash
pip install git+https://github.com/MojtabaMohasel/MicroNAS.git
```

You may also browse other branches and releases in the  
KerasTuner GitHub repository: https://github.com/keras-team/keras-tuner

---

## Quick Introduction

### Import KerasTuner and TensorFlow

```python
import micronas
from tensorflow import keras
import numpy as np
```

### Define a model-building function

Use the `hp` argument to declare tunable hyperparameters:

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(input_shape=(num_features,)))
    model.add(keras.layers.Dense(
        hp.Int("units", min_value=10, max_value=512, step=5),
        activation="relu"))
    model.add(keras.layers.Dense(1, activation="relu"))
    model.compile(loss="mse")
    return model
```

### Initialize the tuner (with memory constraint support)

```python
tuner = micronas.RandomSearch(
    build_model,
    objective="val_loss",
    max_model_size=10000,
    max_consecutive_failed_trials=float("inf"),
    max_trials=10,
)
```

Where:

| Parameter | Meaning |
|---------|--------|
| `objective` | Metric used to select the best model |
| `max_trials` | Max number of model configurations tested |
| `max_model_size` | **Maximum number of trainable parameters allowed** |
| `max_consecutive_failed_trials` | Prevents search termination due to repeated failures |

---

### Run the search and retrieve the best model

```python
num_samples = 1000
num_features = 20
X_train = np.random.randn(num_samples, num_features)
y_train = np.random.randn(num_samples, 1)

tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

best_model = tuner.get_best_models()[0]
```

---

### Monitor memory-violating trials

```python
memory_exceeded_count = 0
total_trials = len(tuner.oracle.trials)

for trial in tuner.oracle.trials.values():
    if trial.metrics.get_best_value("val_loss") == float("inf"):
        memory_exceeded_count += 1

print(f"{memory_exceeded_count} out of {total_trials} trials exceeded the memory constraint")
```

## Memory-Safe Search with `search_atleast`

This fork also introduces the **`search_atleast()`** API.

Unlike the standard `.search()`, which may finish without producing any valid models if many exceed the memory constraint,  
`search_atleast(n, ...)` **guarantees that at least _n feasible models_ are successfully trained.**

Models that exceed the specified `max_model_size` are simply discarded before they are ever built or compiled — ensuring that **only deployable models are generated.**

### Example Usage

```python
import micronas
import tensorflow as tf

def IMUModel(hp):
    # define your tunable model here
    ...
    return model

tuner = micronas.RandomSearch(
    IMUModel,
    objective=micronas.Objective("objective", direction="max"),
    max_trials=50,
    max_model_size=80000,                     # 🚨 memory constraint
    max_consecutive_failed_trials=float("inf"),
    overwrite=True,
)

print(tuner.search_space_summary())

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
)

# Train at least 2 feasible models
tuner.search_atleast(
    min_trials=20,
    x=window_all_X,
    y=window_all_y,
    epochs=20,
    batch_size=7000,
    callbacks=[callback],
    class_weight=class_weights,
    validation_data=(window_all_val_X, window_all_val_y),
)

models = tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.summary()
```

---

## Citation

If this project supports your research, please cite MicroNAS using:

```bibtex
@article{mohasel2025micronas,
  title={Micronas: An automated framework for developing a fall detection system},
  author={Mohasel, Seyed Mojtaba and Sheppard, John and Molina, Lindsey K and Neptune, Richard R and Wurdeman, Shane R and Pew, Corey A},
  journal={arXiv preprint arXiv:2504.07397},
  year={2025}
}
```
