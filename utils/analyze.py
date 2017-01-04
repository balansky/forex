import numpy as np
import talib
from sklearn.externals import joblib


def count_gain_loss(fx_pair, fw_days, bk_days):
    emafw = talib.EMA(fx_pair.cprice, timeperiod=fw_days)
    emabk = talib.EMA(fx_pair.cprice, timeperiod=bk_days)
    gain_in_bull = []
    loss_in_bull = []
    gain_in_bear = []
    loss_in_bear = []
    for i in range(bk_days,fx_pair.num_examples - fw_days):
        diff_max = abs(fx_pair.cprice[i] - max(fx_pair.hprice[i: i + fw_days]))
        diff_min = abs(fx_pair.cprice[i] - min(fx_pair.lprice[i: i + fw_days]))
        if emafw[i + fw_days] > emabk[i]:
            gain_in_bull.append(diff_max)
            loss_in_bull.append(diff_min)
        else:
            gain_in_bear.append(diff_min)
            loss_in_bear.append(diff_max)
    lbull_mean = np.mean(loss_in_bull)
    lbull_std = np.std(loss_in_bull)
    gbull_mean = np.mean(gain_in_bull)
    lbear_mean = np.mean(loss_in_bear)
    lbear_std = np.std(loss_in_bear)
    gbear_mean = np.mean(gain_in_bear)
    return gbull_mean, lbull_mean, lbull_std, gbear_mean, lbear_mean, lbear_std


# def next_signal(fx_pair, fw_days, bk_days,feature_days):
#     gbull_mean, lbull_mean, lbull_std, gbear_mean, lbear_mean, lbear_std = count_gain_loss(fx_pair, fw_days,bk_days)
#     start_idx = fx_pair.num_examples - 1
#     end_idx = fx_pair.num_examples
#
#     pred_X = fx_pair.prepare_X(start_idx, end_idx, feature_days)
#
#     model = joblib.load("models/vcomb")
#     pred = model.predict(pred_X)
#     if pred[0] == 0 :
#         loss_bound = fx_pair.cprice[start_idx] - lbull_mean - 2 * lbull_std
#         gain_bound = fx_pair.cprice[start_idx] + gbull_mean
#     else:
#         loss_bound = fx_pair.cprice[start_idx] + lbear_mean + 2 * lbear_std
#         gain_bound = fx_pair.cprice[start_idx] - gbear_mean
#     return pred[0], gain_bound, loss_bound

def calculate_momentum(fx,idx, bk_days):
    bk_up_diff = fx.hprice[idx - bk_days: idx] - fx.cprice[idx - bk_days: idx]
    bk_dw_diff = fx.cprice[idx - bk_days: idx] - fx.lprice[idx - bk_days: idx]
    bk_up_mean = np.mean(bk_up_diff)
    bk_dw_mean = np.mean(bk_dw_diff)
    bk_up_std = np.std(bk_up_diff)
    bk_dw_std = np.std(bk_dw_diff)
    return bk_up_mean, bk_up_std, bk_dw_mean, bk_dw_std


def calculate_boundary(fx_pair, fw_days, bk_days):
    emafw = talib.EMA(fx_pair.cprice, timeperiod=fw_days)
    emabk = talib.EMA(fx_pair.cprice, timeperiod=bk_days)
    gain_in_bull = []
    loss_in_bull = []
    gain_in_bear = []
    loss_in_bear = []
    for i in range(bk_days,fx_pair.num_examples - fw_days):
        diff_max = abs(fx_pair.cprice[i] - max(fx_pair.hprice[i: i + fw_days]))
        diff_min = abs(fx_pair.cprice[i] - min(fx_pair.lprice[i: i + fw_days]))
        if emafw[i + fw_days] > emabk[i]:
            gain_in_bull.append(diff_max)
            loss_in_bull.append(diff_min)
        else:
            gain_in_bear.append(diff_min)
            loss_in_bear.append(diff_max)
    lbull_mean = np.mean(loss_in_bull)
    lbull_std = np.std(loss_in_bull)
    gbull_mean = np.mean(gain_in_bull)
    lbear_mean = np.mean(loss_in_bear)
    lbear_std = np.std(loss_in_bear)
    gbear_mean = np.mean(gain_in_bear)
    return gbull_mean, lbull_mean + 2 * lbull_std, gbear_mean, lbear_mean + 2 * lbear_std



def next_signal(fx_pair, fw_days, bk_days,feature_days):
    start_idx = fx_pair.num_examples - 1
    end_idx = fx_pair.num_examples
    pred_X = fx_pair.prepare_X(start_idx, end_idx, feature_days)
    gbull, lbull, gbear, lbear = calculate_boundary(fx_pair,fw_days,bk_days)
    bk_up_mean, bk_up_std, bk_dw_mean, bk_dw_std = calculate_momentum(fx_pair,start_idx, bk_days)
    sell_point = fx_pair.cprice[start_idx] + bk_up_mean
    buy_point = fx_pair.cprice[start_idx] - bk_dw_mean
    bull_profit_point = buy_point + 2 * bk_up_mean
    bull_loss_point = buy_point - lbull
    bear_profit_point = sell_point - 2 * bk_dw_mean
    bear_loss_point = sell_point + lbear
    model = joblib.load("models/vcomb")
    pred = model.predict(pred_X)
    if pred[0] == 0 :
        return pred[0], buy_point, bull_profit_point, bull_loss_point
    else:
        return pred[0], sell_point, bear_profit_point, bear_loss_point