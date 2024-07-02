# General Imports
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
import pandas as pd

# Machine Learning Metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# TensorFlow and Keras imports for Neural Network
import tensorflow as tf