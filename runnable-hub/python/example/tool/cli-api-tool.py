#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.apiWorker.worker import Worker as ApiWorker
from runnable_workers.processWorker.worker import Worker as ProcessWorker
from runnable_workers.jinjaWorker.worker import Worker as JinjaWorker
from runnable_workers.toolWorker.worker import Worker as ToolWorker

from runnable_workers.toolWorker.request.toolRequest import ToolRequest
from runnable_workers.toolWorker.request.toolDefine import ToolDefine
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

requestYaml = """
    toolCode: get_domain_ip
    toolVersion: v1
    inputs:
      domain: baidu.com
"""

defineYaml = """
    toolCode: get_domain_ip
    toolVersion: v1
    toolType: API
    setting:
      outputLoads: JSON
      headers: 
        Accept: application/dns-json
      params:
        name: "{{ inputs.domain }}"
        type: A
      url: https://cloudflare-dns.com/dns-query
      method: GET
    inputSpec:
    - name: domain
      type: STRING
      required: true
    outputSpec:
    - name: ip
      type: STRING
    outputsLoads: JSON
    outputTemplate: |
      {"ip": "{{result.Answer[0].data}}"}
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/runnableHub/context"))
    toolWorker = ToolWorker(store=RunnableLocalFileStore("/tmp/runnableHub/tool"))
    runnableHub.registerWorker(ApiWorker())
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(JinjaWorker())
    runnableHub.registerWorker(toolWorker)

    print(runnableHub.workers)

    toolWorker.addTool(ToolDefine.model_validate_json(json.dumps(yaml.safe_load(defineYaml))))

    runnableContext = await runnableHub.executeStart(ToolRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())