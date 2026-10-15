"""Small STRING client used by the compound-commonality workflow.

API specification: https://string-db.org/help/api/
The versioned endpoint can be overridden through STRING_API_BASE.
"""
import os
import time
import requests

BASE_URL = os.environ.get("STRING_API_BASE", "https://version-12-0.string-db.org/api").rstrip("/")

def _request(method, identifiers, **parameters):
    data = {"identifiers": "\r".join(map(str, identifiers)),
            "caller_identity": "DeepHeal", **parameters}
    response = requests.post(f"{BASE_URL}/tsv/{method}", data=data, timeout=120)
    response.raise_for_status()
    time.sleep(1)
    return response.text

def string_map_ids(identifiers, species=9606, limit=1):
    return _request("get_string_ids", identifiers, species=species, limit=limit, echo_query=1)

def string_network(identifiers, species=9606, required_score=700):
    return _request("network", identifiers, species=species, required_score=required_score)
