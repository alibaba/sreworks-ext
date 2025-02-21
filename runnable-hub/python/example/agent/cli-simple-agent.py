#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.agentWorker.worker import Worker as AgentWorker
from runnable_workers.processWorker.worker import Worker as ProcessWorker

from runnable_workers.agentWorker.request.agentRequest import AgentRequest
from runnable_workers.agentWorker.request.agentDefine import AgentDefine
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

defineToolYaml = """
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

defineAgentYaml = """
    agentCode: domain_checker
    agentVersion: v1
    reasoningRunnables:
    - toolCode: uptime
      toolVersion: v1
    reasoningPromptTemplate:
      systemPrompt: ...
      userPrompt: ...
      completionPattern: "Action:{:s}{:action_input}"
      completionParserScript: |      
        @with_pattern(r"\{.*?(?=\n\w+:|\Z)")
        def action_input(text: str) -> dict:
            action = json.loads(text)
            if action["action"] == "Final answer":
              return action["action_input"]
            else:
              return f"Observation: {Runnable.run(action['action_input'])}"

      
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/runnableHub/context"))
    agentWorker = AgentWorker(store=RunnableLocalFileStore("/tmp/runnableHub/agent"))
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(agentWorker)

    print(runnableHub.workers)

    # agentWorker.addTool(ToolDefine.model_validate_json(json.dumps(yaml.safe_load(defineYaml))))

    runnableContext = await runnableHub.executeStart(AgentRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())