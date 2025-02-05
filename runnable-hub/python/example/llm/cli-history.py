#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.llmWorker.worker import Worker as LlmWorker
from workers.llmWorker.request.llmRequest import LlmRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

requestYaml = f"""
    secretKey: {QWEN_SK}
"""
requestYaml += """
    endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    model: qwen-plus
    systemPrompt: You are a helpful assistant.
    userPrompt: 能否再精简一些?
    history:
    - role: user
      content: 请帮我使用python的正则匹配json
    - role: assistant
      content: |

        在 Python 中使用正则表达式（regex）来匹配 JSON 数据并不是一个常见的做法，因为 JSON 是一种结构化的数据格式，通常更适合使用专门的解析库（如 `json` 模块）来处理。然而，如果你确实需要使用正则表达式来匹配 JSON 的某些部分（例如提取特定的键值对），你可以结合 `re` 模块和 `json` 模块来实现。

        以下是一些示例，展示了如何使用正则表达式匹配 JSON 数据的不同部分：

        ### 示例 1：从 JSON 字符串中提取特定键的值

        假设你有一个 JSON 字符串，并且你想提取其中某个键的值。可以使用正则表达式来匹配该键及其对应的值。

        ```python
        import re
        import json

        # 示例 JSON 字符串
        json_str = '{"name": "Alice", "age": 30, "city": "New York"}'

        # 使用正则表达式匹配 "name" 键的值
        pattern = r'"name"\s*:\s*"([^"]*)"'
        match = re.search(pattern, json_str)

        if match:
            print("Name:", match.group(1))
        else:
            print("Key 'name' not found")
        ```

        ### 示例 2：从 JSON 数组中提取所有键值对

        如果你有一个包含多个对象的 JSON 数组，并且想提取每个对象中的某个键的值，可以使用正则表达式来匹配这些键值对。

        ```python
        import re
        import json

        # 示例 JSON 数组字符串
        json_str = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'

        # 使用正则表达式匹配所有的 "name" 键的值
        pattern = r'"name"\s*:\s*"([^"]*)"'
        matches = re.findall(pattern, json_str)

        if matches:
            print("Names found:", matches)
        else:
            print("No names found")
        ```

        ### 示例 3：结合 `json` 模块进行更复杂的操作

        如果你需要更复杂的操作，建议先将 JSON 字符串解析为 Python 对象，然后再进行处理。这样可以避免正则表达式的复杂性和潜在的错误。

        ```python
        import json

        # 示例 JSON 字符串
        json_str = '{"name": "Alice", "age": 30, "city": "New York"}'

        # 解析 JSON 字符串为 Python 字典
        data = json.loads(json_str)

        # 提取 "name" 键的值
        name = data.get('name')
        print("Name:", name)
        ```

        ### 示例 4：匹配 JSON 中的嵌套结构

        如果你有嵌套的 JSON 结构，并且想提取其中的某个键的值，可以使用递归或正则表达式结合解析库。

        ```python
        import re
        import json

        # 示例嵌套 JSON 字符串
        json_str = '{"user": {"name": "Alice", "details": {"age": 30, "city": "New York"}}}'

        # 使用正则表达式匹配嵌套结构中的 "age" 键的值
        pattern = r'"age"\s*:\s*(\d+)'
        match = re.search(pattern, json_str)

        if match:
            print("Age:", match.group(1))
        else:
            print("Key 'age' not found")

        # 或者使用 json 模块解析并提取
        data = json.loads(json_str)
        age = data['user']['details'].get('age')
        print("Age:", age)
        ```

        ### 总结

        虽然正则表达式可以用于匹配 JSON 数据中的某些部分，但在大多数情况下，使用 Python 的 `json` 模块来解析 JSON 数据会更加安全、可靠和高效。正则表达式适用于简单的模式匹配任务，但对于复杂的 JSON 结构，建议使用专业的解析库。

"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(LlmWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(LlmRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())