#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from tools.MapReduce.HDFS import HDFSCluster
from langchain import OpenAI

ai = AI()

#resp = ai.run("47.94.18.251 这是hfds集群的一台master机器，可以直连。帮忙看一下这个hdfs集群有哪些namenode，以及每台namenode机器当前load是多少", llm=OpenAI(temperature=0.2, model_name="gpt-4"), tools=[HDFSCluster], verbose=True)
resp = ai.run("47.94.18.251 这是hfds集群的一台master机器，可以直连。帮忙看一下这个hdfs集群有哪些namenode，并且查看一下active的那台namenode的load是多少", llm=OpenAI(temperature=0.2,model_name="gpt-4"), tools=[HDFSCluster], verbose=True)
print("=== resp ===")
print(resp)
