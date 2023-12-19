import requests

def send_telegram(message, token, chat_id):
    TOKEN = token
    chat_id = chat_id
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    resp = requests.get(url).json()
    return resp
    
def send_photos(file, token, chat_id):
    token = token
    chat_id = chat_id
    method = "sendPhoto"
    params = {'chat_id': chat_id}
    files = {'photo': file}
    url = f"https://api.telegram.org/bot{token}/"
    resp = requests.post(url + method, params, files=files)
    
    return resp

if __name__ == '__main__':
    message = "hello from your telegram bot"
    token = ""
    chat_id = ""
    send_telegram(message, token, chat_id)
    send_photos(open(r"path/to/photo.png", 'rb'))
