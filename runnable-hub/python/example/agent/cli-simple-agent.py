#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.agentWorker.worker import Worker as AgentWorker
from workers.processWorker.worker import Worker as ProcessWorker

from workers.agentWorker.request.agentRequest import AgentRequest
from workers.agentWorker.request.agentDefine import AgentDefine
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    toolCode: get_domain_ip
    toolVersion: v1
    inputs:
      domain: baidu.com
"""

defineYaml = """
    agentCode: simple_uptime
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