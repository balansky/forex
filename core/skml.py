from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import load,forex
import numpy as np
import talib

bk_days = 10
fw_days = 5
feature_days = 10



def _train_model(clf, train_X, train_y, test_X, test_y):
    ml_clf = {"lg": LogisticRegression(C=5,tol=1e-4,max_iter= 1000, n_jobs=-1,solver='lbfgs'),
              "rf": RandomForestClassifier(n_estimators=9, n_jobs=-1),
              "ml": MLPClassifier(max_iter=1000),
              "svc": SVC(C=1)}
    model = ml_clf[clf].fit(train_X,train_y)
    pred_y = model.predict(test_X)
    # print("Coeff: " + str(model.coef_))
    print("Model Score :" + str(model.score(test_X,test_y)))
    print("Confusion Matrix : ")
    print(str(confusion_matrix(test_y,pred_y)))
    return model


def test_eur():
    eur_pair = load.load_fx_pairs(["EURUSD"])[0]
    eur_X, eur_y = eur_pair.prepare(bk_days,fw_days,feature_days)
    _train_model("lg", eur_X[:-1000],eur_y[:-1000],eur_X[-1000:], eur_y[-1000:])


def test_hcomb():
    fx_pairs = load.load_fx_pairs(["EURUSD", "AUDUSD", "CHFJPY", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY",
                                   "GBPUSD", "USDCAD", "USDCHF", "USDJPY"])
    eur_X, eur_y = fx_pairs[0].prepare(bk_days,fw_days,feature_days)
    stack_X = [eur_X]
    for i in range(1, len(fx_pairs)):
        pair_X, pair_y = fx_pairs[i].prepare(bk_days,fw_days,feature_days)
        stack_X.append(pair_X)
    X = np.hstack(tuple(stack_X))
    y = eur_y
    model = _train_model("lg", X[:-1000], y[:-1000], X[-1000:], y[-1000:])
    joblib.dump(model, "models/hcomb")


def test_vcomb():
    # fx_pairs = load.load_fx_pairs(["EURUSD", "AUDUSD", "CHFJPY", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY",
    #                                "GBPUSD", "USDCAD", "USDCHF", "USDJPY"])
    fx_pairs = load.load_fx_pairs(["EURUSD","AUDUSD", "CHFJPY", "EURCHF", "EURGBP", "EURJPY"])
    stack_X = []
    stack_y = []
    eur_X,eur_y = fx_pairs[0].prepare(bk_days,fw_days,feature_days)
    for i in range(1, len(fx_pairs)):
        pair_X, pair_y = fx_pairs[i].prepare(bk_days,fw_days,feature_days)
        stack_X.append(pair_X)
        stack_y.extend(pair_y)
    X = np.vstack(tuple(stack_X))
    y = stack_y
    model = _train_model("lg",X,y,eur_X,eur_y)
    joblib.dump(model,"models/vcomb")


def count_gain_loss(fx_pair):
    emafw = talib.EMA(fx_pair.cprice, timeperiod=fw_days)
    emabk = talib.EMA(fx_pair.cprice, timeperiod=bk_days)
    emafwh = talib.EMA(fx_pair.cprice, timeperiod=fw_days)
    emafwl = talib.EMA(fx_pair.cprice, timeperiod=fw_days)
    gain_in_bull = []
    loss_in_bull = []
    gain_in_bear = []
    loss_in_bear = []
    for i in range(bk_days,fx_pair.num_examples - fw_days):
        diff_max = abs(emabk[i] - emafwh[i + fw_days])
        diff_min = abs(emabk[i] - emafwl[i + fw_days])
        # diff_max = abs(fx_pair.cprice[i] - emafwh[i + fw_days])
        # diff_min = abs(fx_pair.cprice[i] - emafwl[i + fw_days])
        gain_in_bull.append(diff_max)
        loss_in_bull.append(diff_min)
        gain_in_bear.append(diff_min)
        loss_in_bear.append(diff_max)
    lbull_mean = np.mean(loss_in_bull)
    lbull_std = np.std(loss_in_bull)
    gbull_mean = np.mean(gain_in_bull)
    gbull_std = np.std(gain_in_bull)
    lbear_mean = np.mean(loss_in_bear)
    lbear_std = np.std(loss_in_bear)
    gbear_mean = np.mean(gain_in_bear)
    gbear_std = np.std(gain_in_bear)

    #     # diff_max = abs(fx_pair.cprice[i] - max(fx_pair.hprice[i: i + fw_days]))
    #     # diff_min = abs(fx_pair.cprice[i] - min(fx_pair.lprice[i: i + fw_days]))
    #     diff_max = abs(emabk[i] - emafwh[i+fw_days])
    #     diff_min = abs(emabk[i] - emafwl[i+fw_days])
    #     if emafw[i + fw_days] > emabk[i]:
    #         gain_in_bull.append(diff_max)
    #         loss_in_bull.append(diff_min)
    #     else:
    #         gain_in_bear.append(diff_min)
    #         loss_in_bear.append(diff_max)
    # lbull_mean = np.mean(loss_in_bull)
    # lbull_std = np.std(loss_in_bull)
    # gbull_mean = np.mean(gain_in_bull)
    # gbull_std = np.std(gain_in_bull)
    # lbear_mean = np.mean(loss_in_bear)
    # lbear_std = np.std(loss_in_bear)
    # gbear_mean = np.mean(gain_in_bear)
    # gbear_std = np.std(gain_in_bear)
    return gbull_mean, gbull_std ,lbull_mean, lbull_std, gbear_mean, gbear_std ,lbear_mean, lbear_std


def first_gt_index(target_price,price_data):
    idx = 1000
    gt_price = 0
    meet = False
    for i,price in enumerate(price_data):
        if price >= target_price:
            idx = i
            meet = True
            gt_price = price
            break
    return meet,gt_price, idx

def first_ls_index(target_price, price_data):
    idx = 1000
    meet = False
    ls_price = 0
    for i, price in enumerate(price_data):
        if price <= target_price:
            idx = i
            meet = True
            ls_price = price
            break
    return meet, ls_price, idx


def test_return2():
    eur = load.load_fx_pairs(['EURUSD'])[0]
    eur_X, eur_y = eur.prepare(bk_days, fw_days, feature_days)
    model = joblib.load("models/vcomb")
    preds = model.predict(eur_X)
    fx_td = forex.Fx_Trade(eur,preds,bk_days,fw_days)
    start_idx = feature_days + bk_days
    total_gain = 0
    total_trade = 0
    for i, pred in enumerate(preds[:-fw_days]):
        gain = fx_td.get_profit(start_idx+i, i)
        if gain != 0:
            total_gain += gain
            total_trade += 1
    print("Total Gain : " + str(total_gain))
    print("Total Trade : " + str(total_trade))
    print("Gain Per Trade : " + str(total_gain/total_trade))



def test_return():
    eur = load.load_fx_pairs(['EURUSD'])[0]
    gbull_mean, gbull_std, lbull_mean, lbull_std, gbear_mean, gbear_std, lbear_mean, lbear_std = count_gain_loss(eur)
    eur_X, eur_y = eur.prepare(bk_days, fw_days, feature_days)
    model = joblib.load("models/vcomb")
    preds = model.predict(eur_X)
    start_idx = feature_days + bk_days
    emabk = talib.EMA(eur.cprice, timeperiod=bk_days)
    emafw = talib.EMA(eur.cprice, timeperiod=fw_days)
    total_gain = 0
    total_trade = 0
    incorrect = 0
    for i, pred in enumerate(preds[:-fw_days]):
        # bull_gain_bound = emabk[start_idx + i] + gbull_mean
        # bull_loss_bound = emabk[start_idx + i] - lbull_mean - 2*lbull_std
        # bear_gain_bound = emabk[start_idx + i] - gbear_mean
        # bear_loss_bound = emabk[start_idx + i] + lbear_mean + 2*lbear_std
        bk_up_diff = eur.hprice[start_idx + i - bk_days: start_idx + i] - eur.cprice[start_idx + i - bk_days: start_idx + i]
        bk_dw_diff = eur.cprice[start_idx + i - bk_days: start_idx + i] - eur.lprice[start_idx + i - bk_days: start_idx + i]
        bk_up_mean = np.mean(bk_up_diff)
        bk_dw_mean = np.mean(bk_dw_diff)
        bk_up_std = np.std(bk_up_diff)
        bk_dw_std = np.std(bk_dw_diff)
        bull_gain_bound = eur.cprice[start_idx + i] + 0.006
        bull_loss_bound = eur.cprice[start_idx + i] - lbull_mean - 2*lbull_std
        bear_gain_bound = eur.cprice[start_idx + i] - 0.006
        bear_loss_bound = eur.cprice[start_idx + i] + lbear_mean + 2*lbear_std
        # bk_up_bound = eur.cprice[start_idx + i] + bk_up_mean + 2*bk_up_std
        # bk_dw_bound = eur.cprice[start_idx + i] - bk_dw_mean - 2*bk_dw_std
        max_price = max(eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
        min_price = min(eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
        max_idx = np.argmax(eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
        min_idx = np.argmin(eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])

        bk_std = np.std(eur.cprice[start_idx + i - bk_days: start_idx + i])
        bk_mean = np.mean(eur.cprice[start_idx + i - bk_days: start_idx + i])
        # bk_diff = abs(eur.cprice[start_idx + i] - eur.cprice[start_idx + i - 1])
        bk_diff = abs(eur.cprice[start_idx + i] - bk_mean)

        print("Close Price Sequence: " + str(eur.cprice[start_idx + i: start_idx + i + fw_days + 1]))
        print("High Price Sequence: " + str(eur.hprice[start_idx + i: start_idx + i + fw_days + 1]))
        print("Low Price Sequence: " + str(eur.lprice[start_idx + i: start_idx + i + fw_days + 1]))
        if pred == 0 and eur.cprice[start_idx + i] < bull_gain_bound and bk_diff >= 2*bk_std: #and abs(eur.cprice[start_idx + i] - emabk[start_idx + i])*10000 < 30:
            total_trade += 1
            meet_gt, gt_price, gt_idx = first_gt_index(bull_gain_bound,
                                                       eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
            meet_ls, ls_price, ls_idx = first_ls_index(bull_loss_bound,
                                                   eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
            reverse_idx = np.argmax(preds[i: i + fw_days])
            # reverse_idx = 1000
            if reverse_idx == 0: reverse_idx = 1000
            if meet_gt and (gt_idx < ls_idx) and gt_idx <= reverse_idx:
            # if max_price >= bull_gain_bound and (min_price > bull_loss_bound or max_idx < min_idx) and max_idx <=reverse_idx:
                gain = (bull_gain_bound - eur.cprice[start_idx + i])*10000
                total_gain += gain
                print("Bull Gain at day {0} (".format(gt_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(gain))
            # elif min_price <= bull_loss_bound:
            elif meet_ls and ls_idx <= reverse_idx:
                loss = (eur.cprice[start_idx+i] - bull_loss_bound) * 10000
                total_gain -= loss
                print("Bull Loss at day {0} (".format(ls_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(-loss))
            elif reverse_idx != 0 and reverse_idx != 1000:
                rev = (eur.cprice[start_idx + i + reverse_idx] - eur.cprice[start_idx + i]) * 10000
                total_gain += rev
                print("Bull Reverse at day {0} (".format(reverse_idx) + str(eur.cprice[start_idx + i]) + ") : " + str(
                    rev))
            else:
                even = (eur.cprice[start_idx + i + fw_days] - eur.cprice[start_idx + i])*10000
                total_gain += even
                print("Bull Even (" + str(eur.cprice[start_idx + i]) + ") : " + str(even))
            if emafw[start_idx + i + fw_days] > emabk[start_idx + i]:
                print("Prediction: Correct")
            else:
                incorrect += 1
                print("Prediction: Incorrect")
            print("EMA Back: " + str(emabk[start_idx + i]))
            print("EMA Forword: " + str(emafw[start_idx + i + fw_days]))
            print("Loss Bound : " + str(bull_loss_bound))
            print("Gain Bound : " + str(bull_gain_bound))
            print("================================================================================")
        elif pred == 1 and eur.cprice[start_idx + i ] > bear_gain_bound and bk_diff >= 2*bk_std: #and abs(eur.cprice[start_idx + i] - emabk[start_idx + i])*10000 < 30:
            total_trade += 1
            meet_gt, gt_price, gt_idx = first_gt_index(bear_loss_bound,
                                                       eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
            meet_ls, ls_price, ls_idx = first_ls_index(bear_gain_bound,
                                                   eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
            reverse_idx = np.argmin(preds[i: i + fw_days])
            # reverse_idx = 1000
            if reverse_idx == 0: reverse_idx = 1000
            if meet_ls and (ls_idx < gt_idx) and ls_idx <= reverse_idx:
            # if min_price <= bear_gain_bound and (max_price < bear_loss_bound or min_idx < max_idx) and min_idx <= reverse_idx:
                gain = (eur.cprice[start_idx + i] - bear_gain_bound)*10000
                total_gain += gain
                print("Bear Gain at day {0} (".format(ls_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(gain))
            elif meet_gt and gt_idx <= reverse_idx:
            # elif max_price >= bear_loss_bound:
                loss = (bear_loss_bound - eur.cprice[start_idx + i])*10000
                total_gain -= loss
                print("Bear Loss at day {0} (".format(gt_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(-loss))
            elif reverse_idx != 0 and reverse_idx != 1000:
                rev = (eur.cprice[start_idx + i] - eur.cprice[start_idx + i + reverse_idx]) * 10000
                total_gain += rev
                print("Bear Reverse at day {0} (".format(reverse_idx) + str(eur.cprice[start_idx + i]) + ") : " + str(
                    rev))
            else:
                even = (eur.cprice[start_idx + i] - eur.cprice[start_idx + i + fw_days])*10000
                total_gain += even
                print("Bear Even (" + str(eur.cprice[start_idx + i]) + ") : " + str(even))
            if emafw[start_idx + i + fw_days] < emabk[start_idx + i]:
                print("Prediction: Correct")
            else:
                incorrect += 1
                print("Prediction: Incorrect")
            print("EMA Back: " + str(emabk[start_idx + i]))
            print("EMA Forword: " + str(emafw[start_idx + i + fw_days]))
            print("Loss Bound : " + str(bear_loss_bound))
            print("Gain Bound : " + str(bear_gain_bound))
            print("================================================================================")
        else:
            print("================================================================================")
            continue
    print("Total Gain : " + str(total_gain))
    print("Total Trade : " + str(total_trade))
    print("Predict Accuracy :" + str((total_trade - incorrect)/total_trade))
    print("Gain Per Trade : " + str(total_gain/total_trade))



# def test_return():
#     eur = load.load_fx_pairs(["EURUSD"])[0]
#     gbull_mean, gbull_std,lbull_mean, lbull_std, gbear_mean, gbear_std,lbear_mean, lbear_std = count_gain_loss(eur)
#     eur_X, eur_y = eur.prepare(bk_days,fw_days,feature_days)
#     model = joblib.load("models/vcomb")
#     preds = model.predict(eur_X)
#     start_idx = feature_days + bk_days
#     total_gain = 0
#     total_trade = 0
#     for i, pred in enumerate(preds[:-fw_days]):
#         bk_up_diff = eur.hprice[start_idx + i - bk_days: start_idx + i + 1] - eur.cprice[start_idx + i - bk_days: start_idx + i + 1]
#         bk_dw_diff = eur.cprice[start_idx + i - bk_days: start_idx + i + 1] - eur.lprice[start_idx + i - bk_days: start_idx + i + 1]
#         bk_up_mean = np.mean(bk_up_diff)
#         bk_dw_mean = np.mean(bk_dw_diff)
#         bk_up_std = np.std(bk_up_diff)
#         bk_dw_std = np.std(bk_dw_diff)
#         bk_up_bound = eur.cprice[start_idx + i] + bk_up_mean + 2*bk_up_std
#         bk_dw_bound = eur.cprice[start_idx + i] - bk_dw_mean - 2*bk_dw_std
#
#         meet_bk_up, up_idx = first_gt_index(bk_up_bound,eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#         meet_bk_dw, dw_idx = first_ls_index(bk_dw_bound,eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#
#
#
#         # max_price = max(eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#         # min_price = min(eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#         #
#         # max_idx = np.argmax(eur.hprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#         # min_idx = np.argmin(eur.lprice[start_idx + i + 1: start_idx + i + fw_days + 1])
#         print("Close Price Sequence: " + str(eur.cprice[start_idx + i : start_idx + i + fw_days +1]))
#         print("High Price Sequence: " + str(eur.hprice[start_idx + i : start_idx + i + fw_days + 1]))
#         print("Low Price Sequence: " + str(eur.lprice[start_idx + i : start_idx + i + fw_days + 1]))
#         if pred == 0 and meet_bk_up:
#             max_price = max(eur.hprice[start_idx + i + 1 + up_idx: start_idx + i + fw_days + 1])
#             min_price = max(eur.lprice[start_idx + i + 1 + up_idx: start_idx + i + fw_days + 1])
#             max_idx = np.argmax(eur.hprice[start_idx + i + 1 + up_idx: start_idx + i + fw_days + 1])
#             min_idx = np.argmin(eur.lprice[start_idx + i + 1 + up_idx: start_idx + i + fw_days + 1])
#             loss_bound = eur.cprice[start_idx + i] - lbull_mean - 2*lbull_std
#             gain_bound = eur.cprice[start_idx + i] + gbull_mean# - gbull_std
#             if gain_bound <= bk_up_bound: continue
#             total_trade += 1
#             reverse_idx = 1000
#             print("Buying Point at day {0} : ".format(up_idx)  + str(bk_up_bound))
#             # reverse_idx = np.argmax(preds[i : i + fw_days])
#             if reverse_idx == 0: reverse_idx = 1000
#             if max_price >= gain_bound and (min_price > loss_bound or max_idx < min_idx) and max_idx <=reverse_idx:
#                 # gain = (gbull_mean)*10000
#                 gain = (gain_bound - bk_up_bound)*10000
#                 total_gain += gain
#                 print("Bull Gain at day {0} (".format(max_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(gain))
#             elif min_price <= loss_bound and min_idx <= reverse_idx:
#                 # loss = (lbull_mean + 2*lbull_std)*10000
#                 loss = (bk_up_bound - loss_bound)*10000
#                 total_gain -= loss
#                 print("Bull Loss at day {0} (".format(min_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(-loss))
#             # elif reverse_idx != 0 and reverse_idx != 1000:
#             #     # rev = (eur.cprice[start_idx + i + reverse_idx] - eur.cprice[start_idx + i])*10000
#             #     rev = (eur.cprice[start_idx + i + reverse_idx] - bk_up_bound)*10000
#             #     total_gain += rev
#             #     print("Bull Reverse at day {0} (".format(reverse_idx) + str(eur.cprice[start_idx + i]) + ") : " + str(rev))
#             else:
#                 # even = (eur.cprice[start_idx + i + fw_days] - eur.cprice[start_idx + i])*10000
#                 even = (eur.cprice[start_idx + i + fw_days] - bk_up_bound)*10000
#                 total_gain += even
#                 print("Bull Even (" + str(eur.cprice[start_idx + i]) + ") : " + str(even))
#         elif pred == 1 and meet_bk_dw:
#             loss_bound = eur.cprice[start_idx + i] + lbear_mean + 2*lbear_std
#             gain_bound = eur.cprice[start_idx + i] - gbear_mean #+ gbear_std
#             max_price = max(eur.hprice[start_idx + i + 1 + dw_idx: start_idx + i + fw_days + 1])
#             min_price = max(eur.lprice[start_idx + i + 1 + dw_idx: start_idx + i + fw_days + 1])
#             max_idx = np.argmax(eur.hprice[start_idx + i + 1 + dw_idx: start_idx + i + fw_days + 1])
#             min_idx = np.argmin(eur.lprice[start_idx + i + 1 + dw_idx: start_idx + i + fw_days + 1])
#             if gain_bound >= bk_dw_bound: continue
#             total_trade += 1
#             reverse_idx = 1000
#             # reverse_idx = np.argmin(preds[i : i + fw_days])
#             print("Selling Point at day {0} : ".format(dw_idx) + str(bk_dw_bound))
#             if reverse_idx == 0: reverse_idx = 1000
#             if min_price <= gain_bound and (max_price < loss_bound or min_idx < max_idx) and min_idx <= reverse_idx:
#                 # gain = (gbear_mean)*10000
#                 gain = (bk_dw_bound - gain_bound)*10000
#                 total_gain += gain
#                 print("Bear Gain at day {0} (".format(min_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(gain))
#             elif max_price >= loss_bound and max_idx <= reverse_idx:
#                 # loss = (lbear_mean + 2*lbear_std)*10000
#                 loss = (loss_bound - bk_dw_bound)*10000
#                 total_gain -= loss
#                 print("Bear Loss at day {0} (".format(max_idx + 1) + str(eur.cprice[start_idx + i]) + ") : " + str(-loss))
#             # elif reverse_idx != 0 and reverse_idx != 1000:
#             #     # rev = (eur.cprice[start_idx + i] - eur.cprice[start_idx + i + reverse_idx])*10000
#             #     rev = (bk_dw_bound - eur.cprice[start_idx + i + reverse_idx])*10000
#             #     total_gain += rev
#             #     print("Bear Reverse at day {0} (".format(reverse_idx ) + str(eur.cprice[start_idx + i]) + ") : " + str(rev))
#             else:
#                 # even = (eur.cprice[start_idx + i] - eur.cprice[start_idx + i + fw_days])*10000
#                 even = (bk_dw_bound - eur.cprice[start_idx + i + fw_days])*10000
#                 total_gain += even
#                 print("Bear Even (" + str(eur.cprice[start_idx + i]) + ") : " + str(even))
#         else:
#             print("================================================================================")
#             continue
#         print("Loss Bound : " + str(loss_bound))
#         print("Gain Bound : " + str(gain_bound))
#         print("Reverse Index : " + str(reverse_idx))
#         print("================================================================================")
#     print("Total Gain : " + str(total_gain))
#     print("Total Trade : " + str(total_trade))
#     print("Gain Per Trade : "  + str(total_gain/total_trade))


def test_predict():
    eur = load.load_fx_pairs(["EURUSD"])[0]
    gbull_mean, gbull_std, lbull_mean, lbull_std, gbear_mean, gbear_std, lbear_mean, lbear_std = count_gain_loss(eur)
    #gbull_mean, lbull_mean, lbull_std, gbear_mean, lbear_mean, lbear_std = count_gain_loss(eur)
    start_idx = eur.num_examples - 1
    end_idx = eur.num_examples

    pred_X = eur.prepare_X(start_idx, end_idx, feature_days)

    model = joblib.load("models/vcomb")
    pred = model.predict(pred_X)
    if pred[0] == 0 :
        loss_bound = eur.cprice[-1] - lbull_mean - 2 * lbull_std
        gain_bound = eur.cprice[-1] + gbull_mean
    else:
        loss_bound = eur.cprice[-1] + lbear_mean + 2 * lbear_std
        gain_bound = eur.cprice[-1] - gbear_mean
    print("Predict: " + str(pred))
    print("Gain Bound: " + str(gain_bound))
    print("Loss Bound: " + str(loss_bound))


def test(argv):
    global bk_days
    global fw_days
    global feature_days
    bk_days = 10
    fw_days = argv.forward_days
    feature_days = 60
    # test_eur()
    # test_predict()
    # test_vcomb()
    # test_hcomb()
    test_return2()
