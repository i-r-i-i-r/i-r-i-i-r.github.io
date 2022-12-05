import numpy as np
import pandas as pd
import os
import nonlinopt # 非線形最適化
#from selfmadeio.io_json import read_json
from plot import make_plot, make_scatter # 作図
#from selfmadeio.io_csv import save_csv
import matplotlib.pyplot as plt

# 目的関数
def objfun(param,x,y, gain=200):
    W      = calc_W(param, x, y, gain)
    y_calc = calc_y(W, param, x, gain)
    residual = y_calc - y
    rmse = np.sqrt((residual*residual).sum()/len(residual))
    return rmse

# 制約式
def consfun(param):
    cons = []
    return cons

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi(x, param, gain):
    s     = sigmoid(x, param, gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([ones_*(1-s), x*(1-s), x*x*(1-s), ones_*s, x*s]))
    return Phi

# シグモイド関数
def sigmoid(x, x0, gain):
    return 1/(1+np.exp(-gain*(x-x0)))

# 線形回帰式による出力変数の計算
def calc_y(W, param, x, gain):
    Phi = calc_Phi(x, param, gain)
    Y = np.dot(Phi, W)
    return Y

# リッジ回帰における未知係数の計算
def calc_W(param, x, y, gain, alpha=0.02):
    Phi   = calc_Phi(x, param, gain)
    Phi_T = np.transpose(Phi)
    A = np.linalg.pinv( np.dot(Phi_T, Phi)+alpha*np.eye(Phi.shape[1]) )
    B = np.dot(A, Phi_T)
    W = np.dot(B, y)
    return W

def get_plot(dataset, n_deg, intercept_zero):
    #
    """
    # パラメータの最適化
    l_param_init = np.linspace(0, 1, 21)
    fval=1e8
    for param_init in l_param_init:
        param_opt_, fval_, constype = nonlinopt(config, param_init, (x_data,y_data), objfun, consfun )
        if fval>fval_:
            param_opt = param_opt_
            fval = fval_
    W_opt = calc_W(param_opt, x_data, y_data, gain)
    
    # フィッティング
    gain = 200
    y_fit       = calc_y(W_opt,  param_opt,  x_fit, gain)
    y_data_pure = calc_y(W_true, param_true, x_fit, gain)
    
    # 出力
    #print("x0: true={:.2f}, opt={:.2f}".format(param_true, param_opt[0]))
    #print("W_true")
    #print(W_true)
    #print("W_opt")
    #print(W_opt)
    """
    x_data = dataset["x"].values
    phi_data = np.array([[x_i**i_deg for i_deg in range(n_deg,-1,-1)] for x_i in list(x_data)])
    
    x_fit = np.linspace(x_data[0], x_data[-1], 51)
    if intercept_zero:
        if min(x_data)>0:
            x_fit = np.linspace(0, x_data[-1], 51)
        elif max(x_data)<0:
            x_fit = np.linspace(x_data[0], 0, 51)
    phi_fit = np.array([[x_i**i_deg for i_deg in range(n_deg,-1,-1)] for x_i in list(x_fit) ])
    
    y_data = dataset["y"].values
    if intercept_zero:
        coef_vec = np.linalg.lstsq(phi_data[:, :-1], y_data, rcond=None)[0]
        y_fit = np.dot(phi_fit[:, :-1], coef_vec)
        disp_orient = 1
    else:
        coef_vec = np.linalg.lstsq(phi_data, y_data, rcond=None)[0]
        y_fit = np.dot(phi_fit, coef_vec)
        disp_orient = 0
    
    coef_disp_vec0 = ["{:+.2e}".format(c) for c in list(coef_vec)]
    coef_disp_vec0 = [c[0]+" "+c[1:].replace("e+00","") for c in coef_disp_vec0]
    coef_disp_vec1 = []
    for c in coef_disp_vec0:
        if "e" in c:
            if "e+01" in c:
                coef_disp_vec1 += [c.split("e")[0]+r"$\times$10"]
            else:
                coef_disp_vec1 += [c.split("e")[0]+r"$\times$"+r"10$^{"+str(int(c.split("e")[1]))+"}$"]
        else:
            coef_disp_vec1 += [c]
    
    formula0 = [""]
    if n_deg>=2:
        formula0 = [coef_disp_vec1[i] + r"$x^{"+str(n_deg-i)+"}$" for i in range(0, n_deg-1)]
    
    if intercept_zero:
        formula1 = [coef_disp_vec1[-1]+r"$x$ "]
    else:
        formula1 = [coef_disp_vec1[-2]+r"$x$ "+coef_disp_vec1[-1]]
    
    formula2 = list(filter(None, formula0+formula1))
    
    if formula2[0].split("+")[0]=="":
        formula2[0] = formula2[0][1:]
    
    
    formula3 = r'$y$ = '+' '.join(formula2)
    
    #plt.title(formula3)
    #plt.show()
    
    
    fig_name = "fitting_"
    return make_plot([x_data, x_fit], [y_data, y_fit], fig_name,\
              marker=["o", "None"], line_style=["None", "-"], color=["k", "r"], title = formula3)
    
    