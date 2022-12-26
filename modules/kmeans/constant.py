import pandas as pd

_INTRO = """
This app provides an example of **building a simple dashboard using Panel**.\n\n
It demonstrates how to take the output of **k-means clustering on the Penguins dataset** using scikit-learn,
parameterizing the number of clusters and the variables to plot.\n\n
The plot and the table are linked, i.e. selecting on the plot will filter the data in the table.\n\n
The **`x` marks the center** of the cluster.\n\n
This page's content is almost the same as <a style="color: #ff8c00;" href="https://pyscript.net/examples/panel_kmeans.html"> PyScript's example</a>.\n\n
The source code of this page is partly changed the example's one, in order to try file division of Python code and usage of local dataset (not dataset on the Web).
"""

# 表示内容の生成における初期設定
class Constant:
    def __init__(self):
        self.data  = pd.read_csv("penguins.csv").dropna()
        self.cols  = list(self.data.columns)[2:6]
        self.intro = _INTRO

