#!/usr/bin/python
# -*- coding: UTF-8 -*-

from subprocess import Popen, PIPE

def popen(command):
    child = Popen(command, stdin = PIPE, stdout = PIPE, stderr = PIPE, shell = True)
    out, err = child.communicate()
    ret = child.wait()
    return (ret, out.strip(), err.strip())

def kubectl(cmd: str) -> str:
    """the tool can operate the K8s cluster."""
    (ret, out, err) = popen("kubectl "+ cmd)
    return out




