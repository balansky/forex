import json
import pymysql

class db_open():

    def __init__(self, config_path="configs/db_cfg.json"):
        with open(config_path) as fp:
            self.db_cfg = json.load(fp)


    def __enter__(self):
        self.db = Instance(self.db_cfg)
        return self.db


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close_connection()



class Instance():

    def __init__(self, db_cfg):
        self._conn = pymysql.connect(**db_cfg)

    def close_connection(self):
        self._conn.close()

    def select_data(self, sql,dict=False):
        if dict:
            cursor = self._conn.cursor(pymysql.cursors.DictCursor)
        else:
            cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            fetch_data = cursor.fetchall()
        finally:
            cursor.close()
        return fetch_data


    def insert_many_data(self, sql, data):
        cursor = self._conn.cursor()
        try:
            cursor.executemany(sql, data)
            self._conn.commit()
        finally:
            cursor.close()

    def insert_data(self, sql):
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            self._conn.commit()
        finally:
            cursor.close()

    def call_proc(self,proc, args):
        cursor = self._conn.cursor()
        try:
            cursor.callproc(proc,args)
            self._conn.commit()
        finally:
            cursor.close()

