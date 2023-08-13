#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
#from tools.MapReduce.HDFS import HDFSCluster
from hdfscluster import HDFSCluster
from langchain import OpenAI

ai = AI()

resp = ai.run("当前这个集群正常吗？", llm=OpenAI(temperature=0.2, model_name="gpt-4"), tools=[HDFSCluster("47.93.25.211")], verbose=True)
print("=== resp ===")
print(resp)



