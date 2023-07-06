#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
from subprocess import Popen, PIPE

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../../")

from aibond import AI
from langchain import OpenAI

ai = AI()


def popen(command):
    child = Popen(command, stdin = PIPE, stdout = PIPE, stderr = PIPE, shell = True)
    out, err = child.communicate()
    ret = child.wait()
    return (ret, out.strip(), err.strip())

@ai.tool()
def k8sLabel(name: str, kind: str, namespace: str) -> str:
    """This tool can fetch the labels of Kubernetes objects."""
    cmd = "kubectl get " + kind + " " + name + " -n " + namespace + " -o jsonpath='{.metadata.labels}'"
    (ret, out, err) = popen(cmd)
    return out

@ai.tool()
def k8sServiceSelectorList(namespace: str) -> str:
    """This tool can find all services within a namespace in Kubernetes and retrieve the label selectors for each service."""
    cmd = "kubectl get svc -n " + namespace + "  -o jsonpath=\"{range .items[*]}{@.metadata.name}:{@.spec.selector}{'\\n'}{end}\""
    (ret, out, err) = popen(cmd)
    return out

@ai.agent(tools=["k8sLabel", "k8sServiceSelectorList"], llm=OpenAI(temperature=0.2), verbose=True)
def k8sPodServiceFinder(name: str, namespace: str) -> str:
    """This tool can find the services associated with a Kubernetes pod resource."""
    return f"帮我列出 {namespace} 这个ns下所有的service，在这个service list中找出与 pod {name} 的label相关的service，返回的结果只有service的名称即可"


a = ai.run("使用所有的工具去查找sreworks这个ns下 prod-health-health-6cbc46567-s6dqp 这个pod的关联的k8s资源", llm=OpenAI(temperature=0.2), agents=["k8sPodServiceFinder"], verbose=True)
print(a)
