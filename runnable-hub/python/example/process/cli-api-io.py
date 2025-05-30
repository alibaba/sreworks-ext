#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.processWorker.worker import Worker as ProcessWorker
from runnable_workers.processWorker.request.processRequest import ProcessRequest
from runnable_workers.apiWorker.worker import Worker as ApiWorker
from runnable_workers.jinjaWorker.worker import Worker as JinjaWorker
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

requestYaml = """
    jobs:
      api:
        steps:
        - id: params
          jinja:
            data: 
              url: https://cloudflare-dns.com/dns-query
            outputLoads: JSON
            template: |
              {
                "url": "{{ url }}",
                "params":{
                    "name": "baidu.com",
                    "type": "A"
                }
              }
        - id: api
          api: 
            url: ${{ steps.params.outputs.url }}
            outputLoads: JSON
            headers:
              Accept: application/dns-json
            params: ${{ steps.params.outputs.params }} 
    
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(ApiWorker())
    runnableHub.registerWorker(JinjaWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(
        ProcessRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())