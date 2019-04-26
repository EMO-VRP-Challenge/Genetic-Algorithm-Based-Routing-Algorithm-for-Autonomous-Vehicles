import glob
import os
from src import BASE_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def sharing():
    folder = os.path.join(BASE_DIR, 'results', '100req', 'd-1')
    for resultFile in folder:
        filename = os.path.splitext(os.path.basename(resultFile))[0]
        if 'alg1' in filename:
            df = pd.read_csv(filename)

        elif 'alg2' in filename:



