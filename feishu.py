import requests

app_id = 'cli_a7bef044ec77100e'
app_secret = 'pV52Q0aIskAn8nOlrQm9temJ6bhxslqa'

def get_access_token():
    url = f"https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal/"
    payload = {
        "app_id": app_id,
        "app_secret": app_secret
    }
    response = requests.post(url, json=payload)
    data = response.json()
    return data['app_access_token']

import time
import requests

access_token = get_access_token()

def get_chat_list():
    url = f"https://open.feishu.cn/open-apis/chat/v4/list"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        "page_size": 3  # 获取前3个群聊
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    print(data)
    return data['data']['items']

def get_messages_from_chat(chat_id):
    url = f"https://open.feishu.cn/open-apis/message/v4/list"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        "chat_id": chat_id,
        "page_size": 20  # 每次获取20条消息
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data['data']['items']

def fetch_messages():
    chats = get_chat_list()
    for chat in chats:
        chat_id = chat['chat_id']
        messages = get_messages_from_chat(chat_id)
        for message in messages:
            print(message['content'])  # 打印聊天内容

while True:
    fetch_messages()  # 每秒获取消息
    time.sleep(1)  # 等待1秒
