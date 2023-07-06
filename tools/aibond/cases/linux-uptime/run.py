#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
from subprocess import Popen, PIPE

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.Linux.SSH import ssh
from langchain import OpenAI

ai = AI()

resp = ai.run("帮我看看 8.130.35.2 这台机器启动了多长时间了", llm=OpenAI(temperature=0.2), tools=[ssh], verbose=True)
print("=== resp ===")
print(resp)
