"""Fetch the pure-Python 16-bit PNG reader as a verified wheel without altering an environment."""
from pathlib import Path
import hashlib, urllib.request, json
root=Path('/scratch2/mdj1412/datasets/diligent_v64')
data=json.load(urllib.request.urlopen('https://pypi.org/pypi/pypng/0.20220715.0/json'))
item=next(x for x in data['urls'] if x['filename'].endswith('.whl'))
body=urllib.request.urlopen(item['url']).read()
assert hashlib.sha256(body).hexdigest()==item['digests']['sha256']
(root/item['filename']).write_bytes(body)
print(item['filename'],item['digests']['sha256'])
