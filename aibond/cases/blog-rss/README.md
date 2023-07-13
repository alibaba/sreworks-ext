# 利用Python处理博客订阅

## 使用工具
- [Python](../../tools/Programming/Python/)

## 使用方法
```
resp = ai.run("你是一个Python编程专家，帮我总结一下 https://elastic.aiops.work/blog/feed 这个xml的博客订最近有什么更新", llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), tools=[execute_python], verbose=True)
```

## 调用过程

```
> Entering new AgentExecutor chain...
Thought: The user is asking for a summary of updates from an XML blog feed. I will need to fetch the XML from the provided URL and parse it to extract the relevant information. Python has several libraries that can help with this, such as 'requests' for fetching the data and 'xml.etree.ElementTree' for parsing the XML. I'll write a Python script to do this and use the execute_python tool to run it.

Action:
{
  "action": "execute_python",
  "action_input": {
    "code_str": "import requests\nfrom xml.etree import ElementTree as ET\n\n# Fetch the XML data\nresponse = requests.get('https://elastic.aiops.work/blog/feed')\n\n# Parse the XML\nroot = ET.fromstring(response.content)\n\n# Extract the relevant information\nupdates = [{'title': item.find('title').text, 'link': item.find('link').text, 'pubDate': item.find('pubDate').text} for item in root.findall('.//item')]\n\n# Print the updates\nfor update in updates:\n    print(f\"Title: {update['title']}\nLink: {update['link']}\nPublished Date: {update['pubDate']}\n\")",
    "return_context": false
  }
}

Observation: {'stdout': '', 'stderr': 'Traceback (most recent call last):\n  File "/Users/j10k/Projects/aibond/cases/blog-rss/../../tools/Programming/Python/__init__.py", line 22, in execute_python\n    exec(code_str, context)\n  File "<string>", line 15\n    print(f"Title: {update[\'title\']}\n                                   ^\nSyntaxError: EOL while scanning string literal\n', 'context': {}}
Thought:The Python code failed to execute due to a SyntaxError. It seems like the print statement in the code is not properly formatted. I need to correct the print statement and try again.

Action:
{
  "action": "execute_python",
  "action_input": {
    "code_str": "import requests\nfrom xml.etree import ElementTree as ET\n\n# Fetch the XML data\nresponse = requests.get('https://elastic.aiops.work/blog/feed')\n\n# Parse the XML\nroot = ET.fromstring(response.content)\n\n# Extract the relevant information\nupdates = [{'title': item.find('title').text, 'link': item.find('link').text, 'pubDate': item.find('pubDate').text} for item in root.findall('.//item')]\n\n# Print the updates\nfor update in updates:\n    print(f\"Title: {update['title']}\\nLink: {update['link']}\\nPublished Date: {update['pubDate']}\\n\")",
    "return_context": false
  }
}

Observation: {'stdout': 'Title: Being one of the only woman in her computer science program didn’t stop Hannah Mudge from pursuing her dream of becoming a software engineer\nLink: https://www.elastic.co/blog/culture-hannah-mudge-dream-of-becoming-software-engineer\nPublished Date: Fri, 07 Jul 2023 00:00:00 GMT\n\nTitle: Accelerating R&D in pharma with Elasticsearch, ESRE, LLMs, and LangChain — Part 1\nLink: https://www.elastic.co/blog/research-development-pharma-elasticsearch-esre-llms-langchain-1\nPublished Date: Fri, 07 Jul 2023 00:00:00 GMT\n\nTitle: Being one of the only woman in her computer science program didn’t stop Hannah Mudge from pursuing her dream of becoming a software engineer\nLink: https://www.elastic.co/blog/culture-hannah-mudge-dream-of-becoming-software-engineer\nPublished Date: Fri, 07 Jul 2023 00:00:00 GMT\n\nTitle: Accelerating R&D in pharma with Elasticsearch, ESRE, LLMs, and LangChain — Part 1\nLink: https://www.elastic.co/blog/research-development-pharma-elasticsearch-esre-llms-langchain-1\nPublish', 'stderr': '', 'context': {}}
Thought:The Python code executed successfully and returned the recent updates from the XML blog feed. The updates include the title, link, and published date of the blog posts. I will now format this information into a readable format for the user.

Action:
{
  "action": "Final Answer",
  "action_input": "Here are the recent updates from the blog:\n\n1. Title: Being one of the only woman in her computer science program didn’t stop Hannah Mudge from pursuing her dream of becoming a software engineer\nLink: https://www.elastic.co/blog/culture-hannah-mudge-dream-of-becoming-software-engineer\nPublished Date: Fri, 07 Jul 2023 00:00:00 GMT\n\n2. Title: Accelerating R&D in pharma with Elasticsearch, ESRE, LLMs, and LangChain — Part 1\nLink: https://www.elastic.co/blog/research-development-pharma-elasticsearch-esre-llms-langchain-1\nPublished Date: Fri, 07 Jul 2023 00:00:00 GMT"
}


> Finished chain.
=== resp ===
Here are the recent updates from the blog:

1. Title: Being one of the only woman in her computer science program didn’t stop Hannah Mudge from pursuing her dream of becoming a software engineer
Link: https://www.elastic.co/blog/culture-hannah-mudge-dream-of-becoming-software-engineer
Published Date: Fri, 07 Jul 2023 00:00:00 GMT

2. Title: Accelerating R&D in pharma with Elasticsearch, ESRE, LLMs, and LangChain — Part 1
Link: https://www.elastic.co/blog/research-development-pharma-elasticsearch-esre-llms-langchain-1
Published Date: Fri, 07 Jul 2023 00:00:00 GMT
```


