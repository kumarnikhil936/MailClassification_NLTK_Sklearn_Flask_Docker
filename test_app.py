import requests

url = 'http://localhost:5000/'
r = requests.post(url,
                  json={
                      'text':"kundin frau vertragsnummer tierarztrechnungzugesandtrechnung anhang beigefugt"
                  }
                 )

print(r.json())