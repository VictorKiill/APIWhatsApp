import urllib.parse

def enviarMensagem(num, msg):
    msg_final = urllib.parse.quote_plus(msg, safe='')
    url = "https://wa.me/" + num + "?text=" + msg_final

    return url