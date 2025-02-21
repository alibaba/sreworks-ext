#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.apiWorker.worker import Worker as ApiWorker
from runnable_workers.apiWorker.request.apiRequest import ApiRequest
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore
from runnable_hub.queue.redis import RunnableRedisQueueBus

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

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"), queueBus=RunnableRedisQueueBus(host="127.0.0.1", port=6379, db=0))
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(ApiRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())