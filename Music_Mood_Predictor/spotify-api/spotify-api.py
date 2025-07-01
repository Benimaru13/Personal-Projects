import requests
import base64

# Replace with your actual ID and Secret
client_id = '66eb029b579847bab07b1e529cf81516'
client_secret = '1e868543961e4370bd188bb21026bea6'

# Encode credentials
auth_string = f"{client_id}:{client_secret}"
b64_auth_string = base64.b64encode(auth_string.encode()).decode()

# Get token
url = 'https://accounts.spotify.com/api/token'
headers = {
    'Authorization': f'Basic {b64_auth_string}',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {'grant_type': 'client_credentials'}

response = requests.post(url, headers=headers, data=data)
access_token = response.json().get('access_token')
print("Access token:", access_token)
