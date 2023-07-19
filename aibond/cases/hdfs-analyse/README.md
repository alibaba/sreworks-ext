# HDFS集群问题分析排查

## 使用工具
- [HDFS](../../tools/MapReduce/HDFS/)

## 使用方法
```
resp = ai.run("47.94.18.251 这是hfds集群的一台master机器，可以直连。帮忙看一下这个hdfs集群有哪些namenode，并且查看一下active的那台namenode的load是多少", llm=OpenAI(temperature=0.2, model_name="gpt-4"), tools=[HDFSCluster], verbose=True)
```

## 调用过程
> Entering new AgentExecutor chain...
The user is asking for information about the HDFS cluster on a master machine with the IP address 47.94.18.251. They want to know which namenodes are present and what the load is on the active namenode. To get this information, I need to use the HDFSCluster tool. First, I will initialize an instance of the HDFSCluster tool with the provided IP address.

Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "__init__",
    "sub_args": {
      "host": "47.94.18.251",
      "username": "emr-user",
      "password": "password"
    }
  }
}

Observation: {'instance_id': '987940402a9d'}
Thought:
Now that I have initialized the HDFSCluster instance, I can use it to get the namenodes of the HDFS cluster. I will use the 'get_namenodes' sub_func with the instance_id I received from the initialization.

Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "get_namenodes",
    "sub_args": {},
    "instance_id": "987940402a9d"
  }
}

Observation: master-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021 standby
master-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021 standby
master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021 active

Thought:

The HDFS cluster has three namenodes. They are located at master-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, master-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, and master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021. The active namenode is master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021. Now, I need to find the load on the active namenode. I will use the 'exec_command' sub_func with the command 'hdfs dfsadmin -report' to get this information.

Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "exec_command",
    "sub_args": {
      "command": "hdfs dfsadmin -report",
      "host": "master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com"
    },
    "instance_id": "987940402a9d"
  }
}

Observation: {'stdout': 'Configured Capacity: 602926153728 (561.52 GB)\nPresent Capacity: 602791936000 (561.39 GB)\nDFS Remaining: 602791624704 (561.39 GB)\nDFS Used: 311296 (304 KB)\nDFS Used%: 0.00%\nReplicated Blocks:\n\tUnder replicated blocks: 0\n\tBlocks with corrupt replicas: 0\n\tMissing blocks: 0\n\tMissing blocks (with replication factor 1): 0\n\tLow redundancy blocks with highest priority to recover: 0\n\tPending deletion blocks: 0\nErasure Coded Block Groups: \n\tLow redundancy block groups: 0\n\tBlock groups with corrupt internal blocks: 0\n\tMissing block groups: 0\n\tLow redundancy blocks with highest priority to recover: 0\n\tPending deletion blocks: 0\n\n-------------------------------------------------\nLive datanodes (2):\n\nName: 172.16.0.247:9866 (core-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com)\nHostname: core-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com\nDecommission Status : Normal\nConfigured Capacity: 301463076864 (280.76 GB)\nDFS Used: 155648 (152 KB)\nNon DFS Used: 0 (0 B)\nDFS Remaining: 301395812352 (280.70 GB)\nDFS Used%: 0.00%\nDFS Remaining%: 99.98%\nConfigured Cache Capacity: 0 (0 B)\nCache Used: 0 (0 B)\nCache Remaining: 0 (0 B)\nCache Used%: 100.00%\nCache Remaining%: 0.00%\nXceivers: 2\nLast contact: Thu Jul 20 06:02:52 CST 2023\nLast Block Report: Thu Jul 20 03:51:49 CST 2023\nNum of Blocks: 1\n\n\nName: 172.16.0.248:9866 (core-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com)\nHostname: core-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com\nDecommission Status : Normal\nConfigured Capacity: 301463076864 (280.76 GB)\nDFS Used: 155648 (152 KB)\nNon DFS Used: 0 (0 B)\nDFS Remaining: 301395812352 (280.70 GB)\nDFS Used%: 0.00%\nDFS Remaining%: 99.98%\nConfigured Cache Capacity: 0 (0 B)\nCache Used: 0 (0 B)\nCache Remaining: 0 (0 B)\nCache Used%: 100.00%\nCache Remaining%: 0.00%\nXceivers: 2\nLast contact: Thu Jul 20 06:02:52 CST 2023\nLast Block Report: Thu Jul 20 00:10:11 CST 2023\nNum of Blocks: 1\n\n\n', 'stderr': "Warning: Permanently added the RSA host key for IP address '172.16.0.250' to the list of known hosts.\r\n", 'exitStatus': 0}

Thought:
The load on the active namenode is represented by the 'DFS Used%' value in the report. In this case, the load is 0.00%, which means that the namenode is not currently under any load. Now I can provide the user with the information they requested.

Action:
{
  "action": "Final Answer",
  "action_input": "The HDFS cluster has three namenodes. They are located at master-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, master-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, and master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021. The active namenode is master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021 and its load is 0.00%."
}

> Finished chain.
=== resp ===
The HDFS cluster has three namenodes. They are located at master-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, master-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021, and master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021. The active namenode is master-1-3.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com:8021 and its load is 0.00%.


