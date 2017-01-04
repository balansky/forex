from mylibs import database
from utils.forex import Fx_Pair


def load_fx_price(db_ins,fx, days=None, date=None):
    sql = "select price_close,price_high,price_low from forex_one_day where price_pair='{0}'".format(fx)
    if date: sql += " where price_time='{0}'".format(date)
    sql += " order by price_time asc"
    if days: sql += " limit {0}".format(days)
    price_data = db_ins.select_data(sql)
    price_close = []
    price_high = []
    price_low = []
    for i ,d in enumerate(price_data):
        price_close.append(d[0] if d[0] > 0 else price_data[i - 1][0])
        price_high.append(d[1] if d[1] > 0 else price_data[i - 1][1])
        price_low.append(d[2] if d[2] > 0 else price_data[i - 1][2])
    return price_close, price_high, price_low


def load_fx_pairs(pairs, days=None, date=None):
    fx_pairs = []
    with database.db_open() as db:
        for pair in pairs:
            pair_cprice, pair_hprice,pair_lprice = load_fx_price(db, pair, days, date)
            fx_pair = Fx_Pair(pair,pair_cprice,pair_hprice,pair_lprice)
            fx_pairs.append(fx_pair)
    return fx_pairs