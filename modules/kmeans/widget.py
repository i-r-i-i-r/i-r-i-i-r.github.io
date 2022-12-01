import panel as pn
from constant import Constant
from analyze_and_visualize import get_plot, get_clusters

# 表示内容の生成における初期設定
_CONST = Constant()

# GUIの構成要素のクラス
class Widget:
    def __init__(self):
        pn.config.sizing_mode = 'stretch_width'
        self.x          = pn.widgets.Select(name='x', options=_CONST.cols, value='bill_depth_mm')
        self.y          = pn.widgets.Select(name='y', options=_CONST.cols, value='bill_length_mm')
        self.n_clusters = pn.widgets.IntSlider(name='n_clusters', start=1, end=5, value=3)
        self.n_clusters.param.watch(self._update_table, 'value')
        self.plot  = pn.pane.Vega()
        self.table = pn.widgets.Tabulator(pagination='remote', page_size=10)
        self.intro = _CONST.intro
        self._update_table()
        self.plot.object = get_plot(self.x.value, self.y.value, self.table.value)
    
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



