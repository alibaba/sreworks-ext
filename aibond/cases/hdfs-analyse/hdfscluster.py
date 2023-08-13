#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Dict
import paramiko
from paramiko.buffered_pipe import PipeTimeout
import time
import socket

class HDFSCluster():
    """This is a class that can operate on an HDFS cluster."""
    _client = None
    _client_host = None
    def __init__(self, host: str, username: str = "emr-user", password: str = None):
        """Initialize the HDFS cluster by connecting to the host."""
        self._client = paramiko.SSHClient()
        self._client_host = host
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(host, username=username, password=password)

    def exec_command(self, command: str, host: str = None, timeout: int = 5) -> Dict:
        if host == None or self._client_host == host:
            stdin, stdout, stderr = self._client.exec_command(command)
        else:
            if host.find(":") != -1:
                host = host.split(":")[0]
            stdin, stdout, stderr = self._client.exec_command("ssh " +  host +  " '" + command + "'")

        stdout.channel.settimeout(timeout)
        retcode = 999
       
        try:  
            output_stdout = stdout.read().decode('utf-8')
            output_stderr = stderr.read().decode('utf-8')
            retcode  = stdout.channel.recv_exit_status()
        except socket.timeout as e:
            output_stdout = ""
            output_stderr = ""
            while True:
               try:
                  output_stdout += stdout.readline() + "\n"
               except socket.timeout as e:
                  break
            while True:
               try:
                  output_stderr += stderr.readline() + "\n"
               except socket.timeout as e:
                  break

        stdin = None
        stdout = None
        stderr = None

        return {"stdout": output_stdout, "stderr": output_stderr, "exitStatus": retcode}

    def get_namenodes(self) -> str:
        """Get the namenode list of the HDFS cluster."""
        res = self.exec_command("hdfs haadmin -getAllServiceState")
        return res['stdout']

    def hdfs_touchz(self) -> str:
        """Create a test file in HDFS."""
        res = self.exec_command("hdfs dfs -touchz test-file")
        return res

    #def hdfs_cat(self, path: str) -> str:
    #    """Read the file in HDFS and delete the test file."""
    #    res = self.exec_command("hdfs dfs -cat " + path)
    #    self.exec_command("hdfs dfs -rm " + path)
    #    return res['stdout']

    def namenode_log(self, host: str) -> str:
        """get one HDFS cluster namenode's log."""
        res = self.exec_command("cd /mnt/disk1/log/hadoop-hdfs && tail -n 30 hadoop-hdfs-namenode-*.log", host)
        return res['stdout'][-2000:]

    def get_local_disk_free(self, host: str):
        res = self.exec_command("df", host)
        return res['stdout']


if __name__ == "__main__":
    class_instance = HDFSCluster("47.93.25.211")
    resp = class_instance.hdfs_touchz()
    print(resp)
