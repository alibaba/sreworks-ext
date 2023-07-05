#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
from subprocess import Popen, PIPE

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.Kubernetes.Kubectl import kubectl
from langchain import OpenAI

ai = AI()

resp = ai.run("帮我看看这个集群有几个node", llm=OpenAI(temperature=0.2), tools=[kubectl], verbose=True)
print("=== resp ===")
print(resp)
