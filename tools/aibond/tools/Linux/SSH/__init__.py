#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Dict
import paramiko

def ssh(command:str, host: str, username: str = "root", password: str = None) -> Dict:
    """A tool that can connect to a remote server and execute commands to retrieve returned content."""

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(host, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command(command)

    retcode  = stdout.channel.recv_exit_status()
    output_stdout = stdout.read().decode('utf-8')
    output_stderr = stderr.read().decode('utf-8')

    stdin = None
    stdout = None
    stderr = None

    ssh.close()

    return {"stdout": output_stdout, "stderr": output_stderr, "exitStatus": retcode}
   



