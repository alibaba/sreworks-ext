#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.apiWorker.worker import Worker as ApiWorker
from workers.apiWorker.request.apiRequest import ApiRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    url: https://cloudflare-dns.com/dns-query
    outputLoads: JSON
    headers:
        Accept: application/dns-json
    params:
        name: baidu.com
        type: A
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(ApiRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())