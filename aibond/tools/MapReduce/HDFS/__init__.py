#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Dict
import paramiko

class HDFSCluster():
    """This is a class that can operate on an HDFS cluster."""
    _client = None
    _client_host = None
    def __init__(self, host: str, username: str = "emr-user", password: str = None):
        self._client = paramiko.SSHClient()
        self._client_host = host
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(host, username=username, password=password)

    def exec_command(self, command: str, host: str = None) -> Dict:
        if host == None or self._client_host == host:
            stdin, stdout, stderr = self._client.exec_command(command)
        else:
            stdin, stdout, stderr = self._client.exec_command("ssh " +  host +  " '" + command + "'")

        retcode  = stdout.channel.recv_exit_status()
        output_stdout = stdout.read().decode('utf-8')
        output_stderr = stderr.read().decode('utf-8')

        stdin = None
        stdout = None
        stderr = None

        return {"stdout": output_stdout, "stderr": output_stderr, "exitStatus": retcode}

    def get_namenodes(self) -> str:
        res = self.exec_command("hdfs haadmin -getAllServiceState")
        return res['stdout']



