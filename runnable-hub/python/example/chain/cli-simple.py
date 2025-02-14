#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.chainWorker.worker import Worker as ChainWorker
from workers.toolWorker.worker import Worker as ToolWorker
from workers.pythonWorker.worker import Worker as PythonWorker
from workers.llmWorker.worker import Worker as LlmWorker
from workers.processWorker.worker import Worker as ProcessWorker
from workers.jinjaWorker.worker import Worker as JinjaWorker
from workers.apiWorker.worker import Worker as ApiWorker
from workers.chainWorker.request.chainRequest import ChainRequest
from workers.toolWorker.request.toolDefine import ToolDefine
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")


toolDefineYaml = """
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

#       get_domain_ip: get_domain_ip(domain: str) -> str - A tool that can fetch the IP address of a domain.


requestYaml = f"""
    llm:
      endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
      model: qwen-plus
      secretKey: {QWEN_SK}
"""

requestYaml += """
    functions: 
    - type: TOOL
      name: get_domain_ip
      version: v1
      description: |
        A tool that can fetch the IP address of a domain.
      inputDefine:
      - name: domain
        type: STRING
        required: true
    systemPrompt: |
      Respond to the human as helpfully and accurately as possible. You have access to the following tools:

      {{ tool_info }}

      Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

      Valid "action" values: "Final Answer" or ssh

      Provide only ONE action per $JSON_BLOB, as shown:

      ```
      {
        "action": $TOOL_NAME,
        "action_input": $INPUT
      }
      ```

      Follow this format:

      Question: input question to answer
      Thought: consider previous and subsequent steps
      Action:
      ```
      $JSON_BLOB
      ```
      Observation: action result
      ... (repeat Thought/Action/Observation N times)
      Thought: I know what to respond
      Action:
      ```
      {
        "action": "Final Answer",
        "action_input": "Final response to human"
      }
      ```

    userPrompt: |
      Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
      Thought:
      Human: What is the IP address of baidu.com?
    
    onNext: |
      import re
      import json
      from pydantic import BaseModel
      from typing import Optional, Dict
   
      class FunctionCall(BaseModel):
        name: str
        input: Dict|str
      
      class ChainDo(BaseModel):
        message: Optional[str] = None             # Message to be replied
        function: Optional[FunctionCall] = None   # Function call to be executed
        finalAnswer: Optional[str] = None         # Final answer to be returned
        overwriteMessage: Optional[str] = None    # Message to be overwritten

      class ChainContext(BaseModel):
        message: str
        function: Optional[str]
        do: ChainDo = ChainDo()

      def onNext(context: ChainContext):
      
        if context.function is None:
          pattern = re.compile(r'Action:\s*```(.*?)```', re.DOTALL | re.IGNORECASE)
          match = pattern.search(context.message)
          if match:
            json_str = match.group(1).strip()  # 提取内容并去除首尾空白
            try:
                action_data = json.loads(json_str)
                if "final answer" in action_data.get("action","").lower():
                  context.do.finalAnswer = action_data["action_input"]
                elif action_data.get("action_input") is not None:
                  context.do.function = FunctionCall(
                    name=action_data["action"], 
                    input=action_data["action_input"])
            except json.JSONDecodeError:
              raise Exception("json parse fail, content:" + json_str)
          else:
            raise Exception("No action found")
        else:
          context.do.message = "Observation: " + context.function

        return context
      
      
"""

async def main():
    
    toolWorker = ToolWorker(store=RunnableLocalFileStore("/tmp/tool"))
    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ChainWorker())
    runnableHub.registerWorker(toolWorker)
    runnableHub.registerWorker(PythonWorker())
    runnableHub.registerWorker(LlmWorker())
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(JinjaWorker())
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    toolWorker.addTool(ToolDefine.model_validate_json(json.dumps(yaml.safe_load(toolDefineYaml))))

    runnableContext = await runnableHub.executeStart(ChainRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())