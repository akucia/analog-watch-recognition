#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.core.display import display

from watch_recognition.data_preprocessing import (
    load_data,
    load_synthethic_data,
    preprocess_targets,
    unison_shuffled_copies,
)
from watch_recognition.models import export_tflite, get_model

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# In[ ]:


plt.style.use("ggplot")

IMAGE_SIZE = (64, 64)

model = get_model(IMAGE_SIZE)


# In[ ]:


model.summary()


# In[4]:


synth = "./data/analog_clocks/label.csv"

X_train_synth, y_train_synth = load_synthethic_data(
    Path(synth), IMAGE_SIZE, n_samples=500
)


# In[5]:


y_train_synth


# In[6]:


X_train, y_train = load_data(
    Path("./data/watch-time-train/labels.csv"),
    IMAGE_SIZE,
)
X_val, y_val = load_data(
    Path("./data/watch-time-validation/labels.csv"),
    IMAGE_SIZE,
)


# In[7]:


X_train = np.vstack((X_train, X_train_synth))


# In[8]:


y_train = pd.concat((y_train, y_train_synth))


# In[9]:


y_train


# In[10]:


len(y_train), len(X_train)


# In[11]:


len(y_val), len(X_val)


# In[12]:


# y_train['minute'].hist(bins=60)


# In[13]:


y_train["hour"].hist(bins=12)


# In[14]:


# X_train, y_train = unison_shuffled_copies(X_train, y_train)

y_train = preprocess_targets(y_train)
y_val = preprocess_targets(y_val)


# In[15]:


import matplotlib.pyplot as plt

# plt.hist(y_train['minute'], bins=60)


# In[16]:


plt.hist(y_train["hour"], bins=12)


# In[17]:


EPOCHS = 200
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
        )
    ],
)


# In[ ]:


H = model.history
lossNames = [hi for hi in H.history.keys() if "val" not in hi]

(fig, ax) = plt.subplots(len(lossNames), 1, figsize=(16, 9))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    n_steps = len(H.history[l])
    ax[i].plot(np.arange(0, n_steps), H.history[l], label=l)
    ax[i].plot(np.arange(0, n_steps), H.history["val_" + l], label="val_" + l)
    ax[i].legend()
# save the losses figure
# plt.tight_layout()


# In[ ]:


from watch_recognition.reports import generate_report

_ = generate_report(X_train, y_train, model)


# In[ ]:


_ = generate_report(X_val, y_val, model)


# In[ ]:


from watch_recognition.reports import predict_on_image

predict_on_image("example_data/test-image-2.jpg", model)


# In[ ]:


# export_tflite(model, "regression.tflite")


# In[ ]:


x = np.linspace(-3, 3)
v = np.tanh(x) * 2
plt.plot(x, v)


# In[ ]:


x = np.linspace(-3, 3, num=100)
v = np.clip((np.tanh(x) * 2 * 20) + 6, 1, 12)
plt.plot(x, v)
plt.vlines(0, 1, 12)


# In[ ]:


x = np.linspace(-2, 2)
val = np.round((x * 20) + 6)
plt.plot(x, val)


# In[ ]:


# import pandas as pd
# data = pd.read_csv("./data/watch-time-train/labels.csv")
# data.to_csv("./data/watch-time-train/labels.csv")
