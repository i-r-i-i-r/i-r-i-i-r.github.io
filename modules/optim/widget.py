import panel as pn
from constant import Constant
from analyze_and_visualize import get_plot, get_clusters

# 表示内容の生成における初期設定
_CONST = Constant()

# GUIの構成要素のクラス
class Widget:
    def __init__(self):
        pn.config.sizing_mode = 'stretch_width'
        self.dataset1 = pn.widgets.RadioBoxGroup(name= "dataset option", options= ["your dataset (upload)", "example dataset"])
        self.dataset2 = pn.widgets.RadioBoxGroup(name= "select example dataset", options=_CONST.DATASET_EX_NAME, value=_CONST.DATASET_EX_NAME[0])
        self.formula    = pn.widgets.Select(name='formula', options=_CONST.FORMULA_NAME, value=_CONST.FORMULA_NAME[0])
        self.slice      = pn.widgets.RadioBoxGroup(name= "slice", options=["not fixed", "fixed at 0"], value=["not fixed"])
        
        self.n_clusters.param.watch(self._update_table, 'value')
        self.plot  = pn.pane.Vega()
        self.table = pn.widgets.Tabulator(pagination='remote', page_size=10)
        self.intro = _CONST.intro
        self._update_table()
        #self.plot.object = get_plot(self.x.value, self.y.value, self.table.value)
    
    #データがアップロードされたら格納する関数をつくる
    """
    def _update_table(self, event=None):
            self.table.value = get_clusters(self.n_clusters.value, _CONST.data, _CONST.cols)
    
    def _update_filters(self, event=None):
        filters = []
        for k, v in (getattr(event, 'new') or {}).items():
           filters.append(dict(field=k, type='>=', value=v[0]))
           filters.append(dict(field=k, type='<=', value=v[1]))
        self.table.filters = filters
    
    def update_plot(self):
        self.plot.object = get_plot(self.x.value, self.y.value, self.table.value)
        self.plot.selection.param.watch(self._update_filters, 'brush')
    """

