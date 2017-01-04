import numpy as np
import talib


class Tensor_Set():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self.X.shape[0]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.X[start:end], self.y[start:end]



class Fx_Pair():

    def __init__(self, pair_name, close_price, high_price, low_price):
        self.name = pair_name
        self._cprice = np.asarray(close_price, dtype=np.float64)
        self._hprice = np.asarray(high_price, dtype=np.float64)
        self._lprice = np.asarray(low_price, dtype=np.float64)
        assert self._cprice.shape[0] == self._hprice.shape[0] == self._lprice.shape[0]
        self._num_examples = self._cprice.shape[0]


    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other,self.__class__):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    @property
    def cprice(self):
        return self._cprice

    @property
    def hprice(self):
        return self._hprice

    @property
    def lprice(self):
        return  self._lprice

    @property
    def num_examples(self):
        return self._num_examples


    def prepare_X(self, start_idx, end_idx,feature_days):
        X = []
        scalar = np.mean(1/self._cprice)
        for curr_idx in range(start_idx, end_idx):
            row_X = []
            for i in range(0, feature_days):
                rel_prev_cprice = ((self._cprice[curr_idx - i] - self._cprice[curr_idx - i - 1]) /
                                  (self._cprice[curr_idx - i]))*scalar
                rel_curr_hprice = ((self._cprice[curr_idx - i] - self._hprice[curr_idx - i]) /
                                  (self._cprice[curr_idx - i]))*scalar
                rel_curr_lprice = ((self._cprice[curr_idx - i] - self.lprice[curr_idx - i]) /
                                  (self._cprice[curr_idx - i]))*scalar
                row_X.extend([rel_prev_cprice,rel_curr_hprice,rel_curr_lprice])
            X.append(row_X)
        return np.array(X)


    # def prepare_X(self, start_idx, end_idx, feature_days):
    #     X = []
    #     ema10 = talib.EMA(self._cprice, timeperiod=10)
    #     ema15 = talib.EMA(self._cprice, timeperiod=15)
    #     ema20 = talib.EMA(self._cprice, timeperiod=20)
    #     ema25 = talib.EMA(self._cprice, timeperiod=25)
    #     ema30 = talib.EMA(self._cprice, timeperiod=30)
    #     ema35 = talib.EMA(self._cprice, timeperiod=35)
    #     ema40 = talib.EMA(self._cprice, timeperiod=40)
    #     ema45 = talib.EMA(self._cprice, timeperiod=45)
    #     ema50 = talib.EMA(self._cprice, timeperiod=50)
    #     ema55 = talib.EMA(self._cprice, timeperiod=55)
    #     ema60 = talib.EMA(self._cprice, timeperiod=60)
    #     for curr_idx in range(start_idx, end_idx):
    #         curr_price = self._cprice[curr_idx]
    #         rel_ema10 = 0 if (curr_price - ema10[curr_idx])/curr_price >0 else 1
    #         rel_ema15 = 0 if (curr_price - ema15[curr_idx])/curr_price >0 else 1
    #         rel_ema20 = 0 if (curr_price - ema20[curr_idx])/curr_price >0 else 1
    #         rel_ema25 = 0 if (curr_price - ema25[curr_idx])/curr_price >0 else 1
    #         rel_ema30 = 0 if (curr_price - ema30[curr_idx])/curr_price >0 else 1
    #         rel_ema35 = 0 if (curr_price - ema35[curr_idx])/curr_price >0 else 1
    #         rel_ema40 = 0 if (curr_price - ema40[curr_idx])/curr_price >0 else 1
    #         rel_ema45 = 0 if (curr_price - ema45[curr_idx])/curr_price >0 else 1
    #         rel_ema50 = 0 if (curr_price - ema50[curr_idx])/curr_price >0 else 1
    #         rel_ema55 = 0 if (curr_price - ema55[curr_idx])/curr_price >0 else 1
    #         rel_ema60 = 0 if (curr_price - ema60[curr_idx])/curr_price >0 else 1
    #         # X.append([rel_ema10,rel_ema15,rel_ema20, rel_ema25,rel_ema30,rel_ema35,rel_ema40,rel_ema45,
    #         #           rel_ema45,rel_ema50,rel_ema55,rel_ema60])
    #         X.append([
    #                   rel_ema10,rel_ema15,rel_ema60])
    #     return np.array(X)

    def prepare_Y(self, start_idx, end_idx,fw_days, bk_days, is_ts):
        y = []
        emafw = talib.EMA(self._cprice, timeperiod=fw_days)
        emabk = talib.EMA(self._cprice, timeperiod=bk_days)
        for i in range(start_idx, end_idx):
            move_mean = np.mean(self._cprice[i-bk_days: i + 1])
            move_std = np.std(self._cprice[i-bk_days: i + 1])
            move_upper = move_mean + 2*move_std
            move_lower = move_mean - 2*move_std
            if emafw[i + fw_days] > emabk[i]: #and emafw[i + fw_days] >= move_upper:
                y.append(0 if not is_ts else [1,0])
            elif emafw[i + fw_days] <= emabk[i] :#and emafw[i + fw_days] <= move_lower:
                y.append(1 if not is_ts else [0,1])
            # elif emafw[i + fw_days] > emabk[i]:
            #     y.append(2)
            # else:
            #     y.append(3)
            # elif emafw[i + fw_days] > emabk[i]:
            #     y.append(1)
            # else:
            #     y.append(1)
        return y

    # def prepare_Y(self, start_idx, end_idx,fw_days, bk_days, is_ts):
    #     y = []
    #     emafw = talib.EMA(self._cprice, timeperiod=fw_days)
    #     emabk = talib.EMA(self._cprice, timeperiod=bk_days)
    #     for i in range(start_idx, end_idx):
    #         move_mean = np.mean(self._cprice[i-bk_days: i + 1])
    #         move_std = np.std(self._cprice[i-bk_days: i + 1])
    #         move_upper = move_mean + 2*move_std
    #         move_lower = move_mean - 2*move_std
    #         if emafw[i + fw_days] > emabk[i]: #and emafw[i + fw_days] >= move_upper:
    #             y.append(0 if not is_ts else [1,0])
    #         elif emafw[i + fw_days] <= emabk[i] :#and emafw[i + fw_days] <= move_lower:
    #             y.append(1 if not is_ts else [0,1])
    #         # elif emafw[i + fw_days] > emabk[i]:
    #         #     y.append(1)
    #         # else:
    #         #     y.append(1)
    #     return y


    def prepare(self, bk_days, fw_days, feature_days, is_ts=False):
        start_idx = feature_days + bk_days
        end_idx = self._num_examples - fw_days
        self._X = self.prepare_X(start_idx,end_idx, feature_days)
        self._y = self.prepare_Y(start_idx,end_idx,fw_days, bk_days, is_ts)
        return self._X, self._y


class Fx_Trade():

    def __init__(self, fx, preds,bk_days, fw_days):
        self._fx = fx
        self._bk_days = bk_days
        self._fw_days = fw_days
        self._preds = preds
        self._gbull, self._lbull, self._gbear,self._lbear = self._calculate_boundary()


    def _calculate_momentum(self, idx):
        bk_up_diff = self._fx.hprice[idx - self._bk_days: idx] - self._fx.cprice[idx - self._bk_days: idx]
        bk_dw_diff = self._fx.cprice[idx - self._bk_days: idx] - self._fx.lprice[idx - self._bk_days: idx]
        bk_up_mean = np.mean(bk_up_diff)
        bk_dw_mean = np.mean(bk_dw_diff)
        bk_up_std = np.std(bk_up_diff)
        bk_dw_std = np.std(bk_dw_diff)
        return bk_up_mean, bk_up_std, bk_dw_mean,bk_dw_std

    def _calculate_boundary(self):
        emabk = talib.EMA(self._fx.cprice, timeperiod=self._bk_days)
        emafwh = talib.EMA(self._fx.cprice, timeperiod=self._fw_days)
        emafwl = talib.EMA(self._fx.cprice, timeperiod=self._fw_days)
        gain_in_bull = []
        loss_in_bull = []
        gain_in_bear = []
        loss_in_bear = []
        for i in range(self._bk_days, self._fx.num_examples - self._fw_days):
            diff_max = abs(emabk[i] - emafwh[i + self._fw_days])
            diff_min = abs(emabk[i] - emafwl[i + self._fw_days])
            gain_in_bull.append(diff_max)
            loss_in_bull.append(diff_min)
            gain_in_bear.append(diff_min)
            loss_in_bear.append(diff_max)
        lbull_mean = np.mean(loss_in_bull)
        lbull_std = np.std(loss_in_bull)
        gbull_mean = np.mean(gain_in_bull)
        lbear_mean = np.mean(loss_in_bear)
        lbear_std = np.std(loss_in_bear)
        gbear_mean = np.mean(gain_in_bear)
        return gbull_mean, lbull_mean + 2*lbull_std, gbear_mean, lbear_mean + 2*lbear_std

    def _lt_idx(self, tprice, lt_price):
        lt_idx = 99
        for i, price in enumerate(lt_price):
            if price <= tprice:
                lt_idx = i + 1
                break
        return lt_idx

    def _gt_idx(self, tprice,gt_price):
        gt_idx = 99
        for i, price in enumerate(gt_price):
            if price >= tprice:
                gt_idx = i + 1
                break
        return gt_idx

    def _reverse_idx(self, idx, rev_dir):
        if rev_dir == 0:
            reverse_idx = np.argmax(self._preds[idx: idx + self._fw_days + 1])
        else:
            reverse_idx = np.argmin(self._preds[idx: idx + self._fw_days + 1])
        if reverse_idx == 0:
            return 99
        else:
            return reverse_idx

    def _buy_trade(self, idx, pred_idx, buy_point, bull_profit_point, bull_loss_point):
        reverse_idx = self._reverse_idx(pred_idx, 0)
        profit_idx = self._gt_idx(bull_profit_point, self._fx.hprice[idx : idx + self._fw_days + 1])
        loss_idx = self._lt_idx(bull_loss_point, self._fx.lprice[idx: idx + self._fw_days + 1])
        # profit_idx = self._gt_idx(bull_profit_point, self._fx.hprice[idx + 1: idx + self._fw_days + 1]) + 1
        # loss_idx = self._lt_idx(bull_loss_point, self._fx.lprice[idx + 1: idx + self._fw_days + 1]) + 1
        print("Reverse Index : " + str(reverse_idx))
        # if reverse_idx == 1:
        #     return 0, 0
        if profit_idx != 99 and profit_idx < loss_idx and profit_idx <= reverse_idx:
            profit = bull_profit_point - buy_point
            return profit_idx, profit
        elif loss_idx != 99 and loss_idx <= profit_idx and loss_idx <= reverse_idx:
            profit = bull_loss_point - buy_point
            return loss_idx, profit
        elif reverse_idx != 99 and reverse_idx < profit_idx and reverse_idx < loss_idx:
            profit = self._fx.cprice[idx - 1 + reverse_idx] - buy_point
            return reverse_idx, profit
        else:
            profit = self._fx.cprice[idx - 1 + self._fw_days] - buy_point
            return self._fw_days, profit

    def _sell_trade(self, idx, pred_idx, sell_point, bear_profit_point, bear_loss_point):
        reverse_idx = self._reverse_idx(pred_idx, 1)
        profit_idx = self._lt_idx(bear_profit_point, self._fx.lprice[idx : idx + self._fw_days + 1])
        loss_idx = self._gt_idx(bear_loss_point, self._fx.hprice[idx: idx + self._fw_days + 1])
        # profit_idx = self._lt_idx(bear_profit_point, self._fx.lprice[idx + 1: idx + self._fw_days + 1]) + 1
        # loss_idx = self._gt_idx(bear_loss_point, self._fx.hprice[idx + 1: idx + self._fw_days + 1]) + 1

        print("Reverse Index : " + str(reverse_idx))
        # if reverse_idx == 1:
        #     return 0, 0
        if profit_idx != 99 and profit_idx < loss_idx and profit_idx <= reverse_idx:
            profit = sell_point - bear_profit_point
            return profit_idx, profit
        elif loss_idx != 99 and loss_idx <= profit_idx and loss_idx <= reverse_idx:
            profit = sell_point - bear_loss_point
            return loss_idx, profit
        elif reverse_idx != 99 and reverse_idx < profit_idx and reverse_idx < loss_idx:
            profit = sell_point - self._fx.cprice[idx - 1 + reverse_idx]
            return reverse_idx, profit
        else:
            profit = sell_point - self._fx.cprice[idx - 1 + self._fw_days]
            return self._fw_days, profit


    def get_profit(self,idx, pred_idx):
        next_price_idx = idx + 1
        bk_up_mean, bk_up_std, bk_dw_mean, bk_dw_std = self._calculate_momentum(idx)
        sell_point = self._fx.cprice[idx] + bk_up_mean#2*bk_up_std #bk_up_mean
        buy_point = self._fx.cprice[idx] - bk_dw_mean#2*bk_dw_std #bk_dw_mean
        #bull_profit_point = buy_point + bk_up_mean
        bull_profit_point = buy_point + 2*bk_up_std
        bull_loss_point = buy_point - self._lbull
        # bear_profit_point = sell_point - bk_dw_mean
        bear_profit_point = sell_point - 2*bk_dw_std
        bear_loss_point = sell_point + self._lbear
        print("Close Price Sequence: " + str(self._fx.cprice[idx: idx + self._fw_days + 1]))
        print("High Price Sequence: " + str(self._fx.hprice[idx: idx + self._fw_days + 1]))
        print("Low Price Sequence: " + str(self._fx.lprice[idx: idx + self._fw_days + 1]))
        if self._preds[pred_idx] == 0 and self._fx.lprice[next_price_idx] <= buy_point:
            break_idx,profit = self._buy_trade(next_price_idx,pred_idx, buy_point,bull_profit_point,bull_loss_point)
            profit = profit*10000
            print("Bull Condition : ")
            print("Current Price : " + str(self._fx.cprice[idx]))
            print("Buy Point : " + str(buy_point))
            print("Gain Point : " + str(bull_profit_point))
            print("Loss Point : " + str(bull_loss_point))
            print("Profit : " + str(profit))
            print("Break at day " + str(break_idx))
        elif self._preds[pred_idx] == 1 and self._fx.hprice[next_price_idx] >= sell_point:
            break_idx, profit = self._sell_trade(next_price_idx,pred_idx, sell_point, bear_profit_point, bear_loss_point)
            profit = profit*10000
            print("Bear Condition: ")
            print("Current Price: " + str(self._fx.cprice[idx]))
            print("Sell Point : " + str(sell_point))
            print("Gain Point : " + str(bear_profit_point))
            print("Loss Point : " + str(bear_loss_point))
            print("Profit : " + str(profit))
            print("Break at day " + str(break_idx))
        else:
            cond = "Bull" if self._preds[pred_idx] == 0 else "Bear"
            print("{0} Condition : ".format(cond))
            if cond == "Bull":
                print("Buy Point : " + str(buy_point))
            else:
                print("Sell Point : " + str(sell_point))
            print("Not Reach to Point")
            profit = 0
        print("================================================================================")
        return profit



