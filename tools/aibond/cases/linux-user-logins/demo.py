#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.Linux.SSH import ssh
from tools.Linux.Ipinfo import ipinfo
from langchain import OpenAI

ai = AI()

resp = ai.run("帮我分析 8.130.35.2 这台机器最近被来自哪些地方的IP登录过", llm=OpenAI(temperature=0.5, model_name="gpt-4"), tools=[ssh,ipinfo], verbose=True)
print("=== resp ===")
print(resp)
