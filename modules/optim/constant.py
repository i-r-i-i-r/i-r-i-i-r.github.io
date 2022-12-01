import pandas as pd

_INTRO = """
This app provides an example of **fitting (finding an approximate formula)**.\n\n
It demonstrates how to take the output of **k-means clustering on the Penguins dataset** using scikit-learn,
parameterizing the number of clusters and the variables to plot.\n\n
The plot and the table are linked, i.e. selecting on the plot will filter the data in the table.\n\n
The **`x` marks the center** of the cluster.
"""

# 表示内容の生成における初期設定
class Constant:
    def __init__(self):
        self.DATASET_EX = [pd.read_csv("ex"+str(i+1)+".csv") for i in range(3)]
        self.DATASET_EX_NAME = ["example "+str(i+1) for i in range(3)]
        self.FORMULA_NAME = ["1", "2", "3"]
        self.intro = _INTRO

