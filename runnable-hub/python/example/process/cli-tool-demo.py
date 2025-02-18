#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.processWorker.worker import Worker as ProcessWorker
from workers.processWorker.request.processRequest import ProcessRequest
from workers.jinjaWorker.worker import Worker as JinjaWorker
from workers.apiWorker.worker import Worker as ApiWorker
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

requestYaml = """
    outputs: ${{ jobs.call.outputs }}
    jobs:
      call:
        outputs: ${{ steps.result.outputs }}
        steps:
        - id: setting
          jinja:
            data: 
              input:
                domain: baidu.com
            outputLoads: JSON
            template: |
              {
                "url": "https://cloudflare-dns.com/dns-query",
                "method": "GET",
                "headers":{
                  "Accept": "application/dns-json"
                },
                "outputLoads": "JSON",
                "params":{
                  "name": "{{ input.domain }}",
                  "type": "A"
                }
              }
        - id: call
          api: ${{ steps.setting.outputs }}
        - id: result
          jinja:
            data: 
              result: ${{ steps.call.outputs }}
            outputLoads: TEXT
            template: |
              {{result.Answer[0].data}}
              


    
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