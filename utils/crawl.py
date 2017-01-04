import requests
import xml.etree.ElementTree as ET
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class PriceCrawler():

    def __init__(self):
        self.forex_url = "https://rates.fxcm.com/RatesXML"

    def _download_xml(self):
        hdr = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/54.0.2840.71 Safari/537.36",
               'Accept-Language': 'en-US,en;q=0.5',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-encoding': 'gzip, deflate, sdch, br'
               }
        page = requests.get(self.forex_url, headers=hdr, verify=False, timeout=20)
        encoding = page.encoding
        return page.content.decode(encoding)

    def _parse_xml(self, xml):
        currencys = {}
        xmldoc = ET.fromstring(xml)
        rate_path = ".//Rate"
        rate_eles = xmldoc.findall(rate_path)
        for ele in rate_eles:
            curr = ele.get("Symbol")
            bid = ele.find("./Bid").text
            ask = ele.find("./Ask").text
            high = ele.find("./High").text
            low = ele.find("./Low").text
            close ="%.5f" % ((float(bid) + float(ask))/2)
            currencys[curr] = {"close":close,"high":high,"low":low}
        return currencys


    def crawl_price_pairs(self):
        xml = self._download_xml()
        currencys = self._parse_xml(xml)
        return currencys