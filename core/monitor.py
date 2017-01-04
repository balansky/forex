from mylibs.log import Logger
from mylibs.mail import MailServer
import time
import json
from mylibs import database
from datetime import datetime,timedelta
from utils import analyze,load,crawl

logger = Logger()

def save_to_database(pricedata, pricetime):
    sql = "insert into forex_five_minute (price_pair,price_close,price_time) values (%s,%s,%s);"
    insert_data = []
    for key,value in pricedata.items():
        insert_data.append((key,value['close'],pricetime))
    with database.db_open() as db:
        db.insert_many_data(sql, insert_data)


def convert_to_daily(currtime):
    yest = currtime - timedelta(1)
    with database.db_open() as db:
        db.call_proc("convert_to_daily",
        (currtime.strftime("%Y-%m-%d"),yest.strftime("%Y-%m-%d %H:%M"),currtime.strftime("%Y-%m-%d %H:%M")))

def send_mail(title,msg):
    with open("configs/mail_cfg.json") as fp:
        mail_cfg = json.load(fp)
    mail = MailServer(**mail_cfg)
    mail.send(title, msg)


def wait_thirty_seconds():
    curr_time = datetime.now()
    if curr_time.second >= 30:
        sleep_time = 60 - curr_time.second
    else:
        sleep_time = 30 - curr_time.second
    logger.message("Not In Time Range, Sleep (" + str(sleep_time) + ") Seconds")
    time.sleep(sleep_time)
    logger.message("Wake Up")


def crawl_price_data(currtime):
    try:
        crawler = crawl.PriceCrawler()
        logger.message("Reached Time Range, Start To Monitor Price...")
        prices = crawler.crawl_price_pairs()
        save_to_database(prices, currtime.strftime("%Y-%m-%d %H:%M:%S"))
        logger.message("Finshed Monitor Price")
    except Exception as err:
        send_mail("Monitor Error Occurs", str(err))
        time.sleep(60)


def generate_singal(argv):
    eur = load.load_fx_pairs([argv.target_pair])[0]
    pred, entry_point,gain_point, loss_point = analyze.next_signal(eur, argv.forward_days, argv.backward_days, argv.feature_days)
    cond = {0: "Bull", 1: "Bear"}
    res_msg = """
    Current Price: {0}
    Prediction Result: {1}
    Entry Point : {2}
    Gain Point : {3}
    Loss Point : {4}
    """.format(eur.cprice[-1],cond[pred], entry_point, gain_point, loss_point)
    return res_msg


def start(argv):
    while True:
        curr_time = datetime.now()
        crawl_price_data(curr_time)
        if curr_time.hour == 15 and curr_time.minute == 30 and curr_time.second <= 10:
            convert_to_daily(curr_time)
            res_msg = generate_singal(argv)
            send_mail("Purchase Signal", res_msg)
        if curr_time.weekday() == 4 and curr_time.hour == 17:
            time.sleep(48*60*60 + 1)
        wait_thirty_seconds()