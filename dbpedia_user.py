import requests
import lxml.html
import csv
import time
import threading
import re

from urllib.parse import quote_plus



def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_numbers(string):
    string = ''.join([letter for letter in string if not letter.isdigit()])
    return string


CONFIDENCE = "0.4"
THREADS = 10
URI_INDEX = {}
INPUT_FILE_NAME = "Userout200.csv" 
OUTPUT_FILE_NAME = "Users200Extended_c40.csv"
ANNOTATE_URL = "https://api.dbpedia-spotlight.org/en/annotate"

base_params = {
    "confidence": CONFIDENCE,
    "support": "0",
    "spotter": "Default",
    "disambiguator": "Default",
    "policy": "whitelist",
    "types": "",
    "sparql": "",
}

base_headers = {
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
}

kws_headers = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Host": "api.dbpedia-spotlight.org",
    "Origin": "https://demo.dbpedia-spotlight.org",
    "Referer": "https://demo.dbpedia-spotlight.org/",
}

classes_headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Host": "dbpedia.org"
}


# These keywords will be ignored
blacklist_kws = [
    "nomads", "Nomads","NOMADS", "Covid", "Covid-19", "COVID-19", "PCR", "Corona", "CORONA", "COVID", "covid", "covid_19", "covid-19", "pcr"
]

# Read data from csv
def read_data():
    raw_data = []
    with open(INPUT_FILE_NAME, newline="", encoding="utf-8") as data_fp:
        csv_reader = csv.reader(data_fp)
        for row in csv_reader:
            raw_data.append(row)
    return raw_data

# Get classes from keyword
def get_classes(keyword, resource_uri):
    if URI_INDEX.get(resource_uri):
        print(f"Getting classes for uri: {resource_uri} from the URI_INDEX")
        classes = URI_INDEX.get(resource_uri).get("classes")
    else:
        print(f"Getting classes for uri: {resource_uri} from the WEBSITE")
        while True:
            try:
                word_resp = session.get(resource_uri, headers=classes_headers)
                break
            except Exception as req_error:
                print(req_error)
                time.sleep(1)

        # print(word_resp, word_resp.url)
        xml = lxml.html.fromstring(word_resp.content)
        raw_classes = xml.xpath('//a[@rel="rdf:type"]//small[contains(text(), "dbo")]//parent::a/text()')
        classes = [i.lstrip(":") for i in raw_classes]
        #
        # add classes to the index
        URI_INDEX[resource_uri] = {
            "keyword": keyword,
            "classes": classes
        }
    #
    print(f"keyword - {keyword} and classes - {classes}")
    return classes


def get_kws_data(description_txt):
    kws_dict = {}
    print(f"Getting kws for txt: {description_txt}")
    emoji_cleaned_txt = remove_emoji(description_txt)
    cleaned_text = remove_numbers(emoji_cleaned_txt)
    print(cleaned_text)
    if not cleaned_text:
        return kws_dict
    params = base_params.copy()
    params["text"] = cleaned_text
    while True:
        try:
            kws_response = session.get(ANNOTATE_URL, params=params, headers=kws_headers)
            break
        except Exception as req_error:
            print(req_error)
            time.sleep(1)
        #
    data_dict = kws_response.json()
    keywords_data = data_dict["Resources"]
    for i in keywords_data:
        kw = i["@surfaceForm"]
        resource_uri = i["@URI"]
        if kw.lower() not in blacklist_kws:
            kws_dict[kw] = resource_uri
            # print(kw, resource_uri)
        else:
            print(f"{kw} is a blacklisted kw")
        # break
    return kws_dict



session = requests.Session()
session.headers.update(base_headers)


raw_data = read_data()
kws_col_index = 0
classes_col_index = 0

headers = raw_data.pop(0)

# raw_data = raw_data[:15]
total_data = len(raw_data)
processed_data = []


def worker():
    global raw_data, kws_col_index, classes_col_index
    while raw_data:
        try:
            row = raw_data.pop()
            print("\n##########################################################################")
            print(f"Remaining Status: {len(raw_data)}/{total_data}")
            # print(row)
            # thumbnail, tags, description
            _thumbs, _tags, description = row[:3]
            kws_data = get_kws_data(description)
            current_keywords = []
            current_classes = []

            # keys are keywords, and values are resource uri
            for kw, uri in kws_data.items():
                current_keywords.append(kw)
                current_classes = current_classes + get_classes(kw, uri)
            
            if len(current_keywords) > kws_col_index:
                kws_col_index = len(current_keywords)
            
            if len(current_classes) > classes_col_index:
                classes_col_index = len(current_classes)

            row.append(list(set(current_keywords)))
            row.append(list(set(current_classes)))
            processed_data.append(row)
            print("##########################################################################\n")
        except Exception as worker_exp:
            print(worker_exp)
            # raw_data.append(row)
    return None



threads = []
for i in range(THREADS):
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    threads.append(th)

[th.join() for th in threads]






# format data before writing to csv
for row in processed_data:
    if len(row) > 3:
        cls_list = row.pop(-1)
        kw_list = row.pop(-1)
        for i in range(kws_col_index):
            try:
                row.append(kw_list[i])
            except:
                row.append("")
        #
        for i in range(classes_col_index):
            try:
                row.append(cls_list[i])
            except:
                row.append("") 

# Adding headers
for i in range(kws_col_index):
    if i == 0:
        headers.append("Keywords")
    else:
        headers.append("")   

for i in range(classes_col_index):
    if i == 0:
        headers.append("Classes")
    else:
        headers.append("") 

# 
processed_data.insert(0, headers)

with open(OUTPUT_FILE_NAME, "w", newline="", encoding="utf-8") as save_fp:
    csv_writer = csv.writer(save_fp)
    for row in processed_data:
        csv_writer.writerow(row)