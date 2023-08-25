#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.Programming.Python import execute_python
from langchain import OpenAI

ai = AI()

resp = ai.run("你是一个Python编程专家，帮我总结一下 https://elastic.aiops.work/blog/feed 这个xml的博客订最近有什么更新", llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), tools=[execute_python], verbose=True)
print("=== resp ===")
print(resp)
