from __future__ import annotations

import base64
from pathlib import Path

import requests

with Path("fish.jpg").open("rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {"img": {"data": imgstr}}}
resp = requests.post("http://127.0.0.1:8000/classify", json=body)
print(resp.json())
