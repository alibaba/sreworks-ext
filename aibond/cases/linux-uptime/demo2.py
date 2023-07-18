#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.Linux.SSH import SshClient
from langchain import OpenAI

ai = AI()

print(SshClient)

#resp = ai.run("帮我看看 8.130.35.2 这台机器启动了多久了", llm=OpenAI(temperature=0.2), tools=[], verbose=True)
resp = ai.run("帮我看看 8.130.35.2 这台机器启动了多久了", llm=OpenAI(temperature=0.2), tools=[SshClient], verbose=True)
print("=== resp ===")
print(resp)
