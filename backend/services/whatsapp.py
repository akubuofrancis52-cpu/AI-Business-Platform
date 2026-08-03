import os
import requests

def send_whatsapp_message(phone_number, message_text):
    """
    Send a WhatsApp message using Meta's Cloud API.
    """
    token = EAALlEdzZCF0wBSMZCvNGlblBZCsKps98YMkfNXwTxuf1KzRXalZA1ZCAY6Cm3qmWkT0cpSBAhfiaV8mVWVCgGWZC6agWJhgJoJGVcauUn4P89WGNd0G1uMvskjfASr8nImZAEA6RJ3lzabdAqfivURZATjVLHgtn44WDnenX6DqJiqcRab0ZAh8icSNZCUZBhuD2F3pKSXL5h7Gvyki67WKScLgTxzM5wAVh50HWZBXPBkv6PXmkGRZCg0IHH0x9V2IXNtEUQ4pJq9QOjIqgqg1zZAYP2x8CpY
    phone_number_id = 1031303756736279

    if not token or not phone_number_id:
        print("WhatsApp credentials (token or phone number ID) missing from environment variables.")
        return False

    url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return True
        else:
            print(f"Failed to send WhatsApp message: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending WhatsApp message: {str(e)}")
        return False