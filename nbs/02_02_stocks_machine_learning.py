import numpy as np

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
InteractiveShell.ast_node_interactivity = "all"

%load_ext autoreload
%autoreload 2


!pip install -Uqq convertdate pystan==2.19.1.1 prophet


import os
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field


%matplotlib inline
pd.set_option('display.max_rows', 500)


COUNTRY = "malaysia"
DATASET_FOLDER = "nbs/test_data"
TARGET_DATASET_FOLDER = f"{DATASET_FOLDER}/{COUNTRY}"
STOCKS_DATASET = f"{TARGET_DATASET_FOLDER}/stocks.csv"
STOCKS_INFO_DATASET = f"{TARGET_DATASET_FOLDER}/stocks_info.csv"
STOCKS_FINANCE_DATASET = f"{TARGET_DATASET_FOLDER}/stocks_finance.csv"
STOCKS_DIVIDENDS_DATASET = f"{TARGET_DATASET_FOLDER}/stocks_dividends.csv"
STOCKS_SELECTED = f"{TARGET_DATASET_FOLDER}/stocks_selected.csv"


def read_csv(file):
    if not os.path.isfile(file):
        return None
    return pd.read_csv(file)

def save_csv(df, file_name):
    df.to_csv(file_name, header=True, index=False)


df_stocks_selected = read_csv(STOCKS_SELECTED)
display(df_stocks_selected.head(10))
# display(df_stocks_selected.info())



