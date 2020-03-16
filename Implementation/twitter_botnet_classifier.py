# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:24:26 2020

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Book1.csv', encoding='latin-1')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,19]

