# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
import os
from hashlib import md5

# Set your own appid/appkey.
appid = os.getenv('BAIDU_APP_ID')
appkey = os.getenv('BAIDU_API_KEY')

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


def trans(text, from_lang='en', to_lang='zh'):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers, proxies={"no_proxy": "baidu.com"})
    result = r.json()

    # Show response
    print(json.dumps(result, indent=4, ensure_ascii=False))
