#!/usr/bin/env python3

import asyncio
from json import tool
import os
import sys
from urllib import request
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.agentWorker.worker import Worker as AgentWorker
from runnable_workers.processWorker.worker import Worker as ProcessWorker
from runnable_workers.toolWorker.worker import Worker as ToolWorker
from runnable_workers.pythonWorker.worker import Worker as PythonWorker
from runnable_workers.llmWorker.worker import Worker as LlmWorker
from runnable_workers.processWorker.worker import Worker as ProcessWorker
from runnable_workers.jinjaWorker.worker import Worker as JinjaWorker
from runnable_workers.apiWorker.worker import Worker as ApiWorker
from runnable_workers.chainWorker.worker import Worker as ChainWorker

from runnable_workers.agentWorker.request.agentRequest import AgentRequest
from runnable_workers.agentWorker.request.agentChainTemplate import AgentChainTemplate
from runnable_workers.llmWorker.request.llmSetting import LlmSetting
from runnable_workers.agentWorker.request.agentDefine import AgentDefine
from runnable_workers.toolWorker.request.toolDefine import ToolDefine
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

defineToolYaml = """
    toolCode: get_domain_ip
    toolVersion: v1
    toolType: API
    description: |
      This tool is used to get the IP address of a domain.
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


defineAgentYaml = """
    agentCode: domain_simple_checker
    agentVersion: v1
    prerun: 
      outputs: ${{ jobs.call.outputs }}
      jobs:
        call:
          outputs: ${{ steps.toolCall.outputs.ip }}
          steps:
          - id: toolCall
            tool:
              toolCode: get_domain_ip
              toolVersion: v1
              inputs:
                domain: "${{ inputs.prompt }}"
    inputDefine:
    - name: prompt
      type: STRING
      required: true
    functions:
    - type: TOOL
      name: get_domain_ip
      version: v1
    instruction: |
      你是一个检查域名的工具，输入一个域名，返回一个ip
      
"""

requestYaml = """
    agentCode: domain_simple_checker
    agentVersion: v1
    inputs:
      prompt: "www.baidu.com"
"""


async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/runnableHub/context"))
    toolWorker = ToolWorker(store=RunnableLocalFileStore("/tmp/runnableHub/tool"))
    agentWorker = AgentWorker(store=RunnableLocalFileStore("/tmp/runnableHub/agent"), toolWorker=toolWorker)
    
    toolWorker.addTool(ToolDefine.model_validate_json(json.dumps(yaml.safe_load(defineToolYaml))))
    agentWorker.addAgent(AgentDefine.model_validate_json(json.dumps(yaml.safe_load(defineAgentYaml))))

    runnableHub.registerWorker(agentWorker)
    runnableHub.registerWorker(toolWorker)

    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(PythonWorker())
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(JinjaWorker())
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(AgentRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())