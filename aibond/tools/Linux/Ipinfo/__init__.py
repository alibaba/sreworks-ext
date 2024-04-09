import requests
from typing import Dict

def ipinfo(ip: str) -> Dict:
    """The tool "ipinfo" can provide queries for information such as the geographical location of an IP address."""
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    return response.json()


