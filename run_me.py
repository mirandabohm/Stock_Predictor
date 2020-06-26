#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np
from tensorflow.keras.models import load_model

# Set parameters 
model = load_model('model.h5')

prediction = model.predict()


