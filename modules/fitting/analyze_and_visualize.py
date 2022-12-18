import numpy as np
import pandas as pd
import os
import nonlinopt # 非線形最適化
from plot import make_plot # 作図
from io_json import read_json
"""
import json
import collections as cl

def read_json(file_name:str) -> dict: # jsonファイルの読み込み
    decoder = json.JSONDecoder(object_pairs_hook=cl.OrderedDict)
    with open(file_name) as conf_file:
        data = decoder.decode(conf_file.read())
    return data

config_opt = read_json("C:\\Users\\irita\\Desktop\\development\\git-pages\\i-r-i-i-r.github.io\\modules\\optim\\config_opt.json")
dataset=pd.read_csv("C:\\Users\\irita\\Desktop\\development\\git-pages\\i-r-i-i-r.github.io\\data\\optim\\example 3.csv")
"""

# 係数の配列から表示用文字列のリストを生成
def coef2str(coef_vec):
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
    return coef_disp_vec1

# 多項式近似
def polyfit_(x_data, y_data, x_fit, intercept_zero, n_deg):
    # 計算
    phi_data = np.array([[x_i**i_deg for i_deg in range(n_deg,-1,-1)] for x_i in list(x_data)])
    
    if intercept_zero:
        if min(x_data)>0:
            x_fit = np.linspace(0, x_data[-1], 51)
        elif max(x_data)<0:
            x_fit = np.linspace(x_data[0], 0, 51)
    phi_fit = np.array([[x_i**i_deg for i_deg in range(n_deg,-1,-1)] for x_i in list(x_fit) ])
    
    if intercept_zero:
        coef_vec = np.linalg.lstsq(phi_data[:, :-1], y_data, rcond=None)[0]
        y_fit = np.dot(phi_fit[:, :-1], coef_vec)
    else:
        coef_vec = np.linalg.lstsq(phi_data, y_data, rcond=None)[0]
        y_fit = np.dot(phi_fit, coef_vec)
    
    # 式（表示用）
    coef_disp_vec1 = coef2str(coef_vec) # 係数の配列から表示用文字列のリストを生成
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
    if n_deg>=6:
        formula3=formula3.split("$x^{4}$")[0]+r"$x^{4}$"+ "\n"+formula3.split("$x^{4}$")[1]
    if n_deg>=9:
        formula3=formula3.split("$x^{7}$")[0]+r"$x^{7}$"+ "\n"+formula3.split("$x^{7}$")[1]
    
    return (y_fit, formula3)


# 目的関数
def objfun(param,x,y,calc_Phi, gain=200):
    W      = calc_W(param, x, y, gain, calc_Phi)
    y_calc = calc_y(W, param, x, gain, calc_Phi)
    residual = y_calc - y
    rmse = np.sqrt((residual*residual).sum()/len(residual))
    return rmse

# 制約式
def consfun(param):
    cons = []
    return cons

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi_11(x, param, gain):
    s     = sigmoid(x, param, gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([x*(1-s), ones_*(1-s), x*s, ones_*s]))
    return Phi

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi_21(x, param, gain):
    s     = sigmoid(x, param, gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([x*x*(1-s), x*(1-s), ones_*(1-s), x*s, ones_*s]))
    return Phi

# 重回帰分析における入力変数の情報を持つ行列の作成
def calc_Phi_12(x, param, gain):
    s     = sigmoid(x, param, gain)
    ones_ = np.ones_like(x)
    Phi = np.transpose(np.vstack([x*(1-s), ones_*(1-s), x*x*s, x*s, ones_*s]))
    return Phi

# シグモイド関数
def sigmoid(x, x0, gain):
    return 1/(1+np.exp(-gain*(x-x0)))

# 線形回帰式による出力変数の計算
def calc_y(W, param, x, gain, calc_Phi):
    Phi = calc_Phi(x, param, gain)
    Y = np.dot(Phi, W)
    return Y

# リッジ回帰における未知係数の計算
def calc_W(param, x, y, gain, calc_Phi, alpha=0.02):
    Phi   = calc_Phi(x, param, gain)
    Phi_T = np.transpose(Phi)
    A = np.linalg.pinv( np.dot(Phi_T, Phi)+alpha*np.eye(Phi.shape[1]) )
    B = np.dot(A, Phi_T)
    W = np.dot(B, y)
    return W


def devided_fit(x_data, y_data, x_fit, n_deg):
    config = read_json("config_opt.json") 
    n_deg =(1,1)# 仮
    if n_deg ==(1,1):
        calc_Phi = calc_Phi_11
    elif n_deg ==(2,1):
        calc_Phi = calc_Phi_21
    elif n_deg ==(1,2):
        calc_Phi = calc_Phi_12
    
    l_param_init = np.linspace(min(x_data), max(x_data), 21)
    fval=1e8
    # パラメータの最適化
    for param_init in l_param_init:
        param_opt_, fval_, constype = nonlinopt.nonlinopt(config, param_init, (x_data,y_data,calc_Phi), objfun, consfun )
        if fval>fval_:
            param_opt = param_opt_
            fval = fval_
    
    # フィッティング
    gain = 200
    W_opt = calc_W(param_opt, x_data, y_data, gain, calc_Phi)
    y_fit = calc_y(W_opt,  param_opt,  x_fit, gain, calc_Phi)
    
    # 仮
    coef_disp_vec1 = coef2str(W_opt)
    param_disp     = coef2str(param_opt)[0].split("+ ")[-1]
    if n_deg ==(1,1):
        formula1 = r"$y = $" + coef_disp_vec1[0].split("+ ")[-1] + r"$x$ " + coef_disp_vec1[1]
        formula2 = r"$y = $" + coef_disp_vec1[2].split("+ ")[-1] + r"$x$ " + coef_disp_vec1[3]
        case1    = r"   $(x\leq$" + param_disp + ")"
        case2    = r"   $(x>$"    + param_disp + ")"
        formula = formula1 + case1 + "\n" + formula2 + case2
    elif n_deg ==(2,1):
        formula = ""
    elif n_deg ==(1,2):
        formula = ""
    
    return (y_fit, formula)


def get_plot(dataset, n_deg, intercept_zero, formula_name):
    x_data = dataset["x"].values
    y_data = dataset["y"].values
    x_fit = np.linspace(min(x_data), max(x_data), 51)
    
    if formula_name=="polyfit":
        y_fit, formula = polyfit_(x_data, y_data, x_fit, intercept_zero, n_deg)
    elif formula_name=="devided line":
        y_fit, formula = devided_fit(x_data, y_data, x_fit, n_deg)
    
    return (make_plot([x_data, x_fit], [y_data, y_fit], "_",\
              marker=["o", "None"], line_style=["None", "-"], color=["k", "r"], title=formula), formula)
    
    