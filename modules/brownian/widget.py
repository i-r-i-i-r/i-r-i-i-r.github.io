# https://panel.holoviz.org/reference/panes/Matplotlib.html
import io
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
from analyze_and_visualize import get_plot


# 表示内容の生成における初期設定
_INTRO = """
This app provides an example of **fitting (finding a good approximation formula)**.\n\n
Use Python to show how well each formula fits the data.
"""

class Option:
    pass


# GUIの構成要素のクラス
class Widget:
    def __init__(self):
        self.DATASET_EX_NAME = ["example "+str(i+1) for i in range(3)]
        self.DATASET_EX = [pd.read_csv(name+".csv") for name in self.DATASET_EX_NAME]
        self.FORMULA_NAME = ["polyfit", "polygonal line", "polygonal curve"]
        self.INTRO = _INTRO
        
        pn.config.sizing_mode = 'stretch_width'
        self.option = Option()
        self.option.N_particles = pn.widgets.IntSlider(name="No. of particles", start=1, end=100, value=10)
        self.option.Temperature = pn.widgets.IntSlider(name="Temperature", start=0, end=100, value=25)
        self.plot = pn.pane.Matplotlib()
        #self.formula_disp = pn.pane.LaTeX('\(F(\omega) = \cfrac{1}{\sqrt{2\pi}}\int_{-\infty}^{+\infty}f(t)e^{i\omega t}dt\)', style={'font-size': '18pt'})
        self.plot.param.trigger('object')
        self.update_plot()
        
        #self.option.data_1.param.watch(self._update_dataset, 'value')
    
    
    #データがアップロードされたら格納する関数をつくる
    def _update_dataset(self, event=None):
        if self.option.data_1.value == "example dataset":
            self.dataset_name = self.option.data_2.value
            self.dataset = self.DATASET_EX[int(self.dataset_name.split(" ")[1]) - 1]
        if self.option.data_1.value == "your dataset" and self.uploaded:
            self.dataset_name = "users data"
            buffer = io.StringIO(self.path.decode("utf-8-sig"))
            self.dataset = pd.read_csv(buffer)
            #self.option.data_2    = pn.widgets.RadioBoxGroup(name= "select example dataset", options=["111","222","333"])
    
    def _update_formula(self, event=None):
        self.polyfit_deg = self.option.polyfit_deg.value
        self.intercept_zero = self.option.intercept.value
    
    # グラフの更新
    def update_plot(self):
        self._update_dataset()
        self._update_formula()
        self.plot.object, _ = get_plot(self.dataset, self.polyfit_deg, self.intercept_zero)
        #self.formula_disp.object = ""
        


