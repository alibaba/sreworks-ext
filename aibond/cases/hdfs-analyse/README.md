# HDFS集群问题分析排查

## 使用工具
- [HDFS](../../tools/MapReduce/HDFS/)

## 使用方法
```
resp = ai.run("47.94.18.251 这是hfds集群的一台master机器，可以直连。帮忙看一下这个hdfs集群有哪些namenode，以及每台namenode机器当前load是多少", llm=OpenAI(temperature=0.2, model_name="gpt-4"), tools=[HDFSCluster], verbose=True)
```

## 调用过程
```
> Entering new AgentExecutor chain...
Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "__init__",
    "sub_args": {
      "host": "47.94.18.251",
      "username": "emr-user"
    }
  }
}

Observation: {'instance_id': '7963257e0311'}
Thought:The HDFSCluster instance has been successfully initialized with the provided host and username. Now, I need to execute a command to find out the namenodes in the HDFS cluster and their current load. The command to list namenodes is "hdfs dfsadmin -report" and the command to check the load on a machine is "uptime". However, I can only execute one command at a time. So, I will first execute the command to list the namenodes.
Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "exec_command",
    "sub_args": {
      "command": "hdfs dfsadmin -report"
    },
    "instance_id": "7963257e0311"
  }
}

Observation: {'stdout': 'Configured Capacity: 602926153728 (561.52 GB)\nPresent Capacity: 602791936000 (561.39 GB)\nDFS Remaining: 602791624704 (561.39 GB)\nDFS Used: 311296 (304 KB)\nDFS Used%: 0.00%\nReplicated Blocks:\n\tUnder replicated blocks: 0\n\tBlocks with corrupt replicas: 0\n\tMissing blocks: 0\n\tMissing blocks (with replication factor 1): 0\n\tLow redundancy blocks with highest priority to recover: 0\n\tPending deletion blocks: 0\nErasure Coded Block Groups: \n\tLow redundancy block groups: 0\n\tBlock groups with corrupt internal blocks: 0\n\tMissing block groups: 0\n\tLow redundancy blocks with highest priority to recover: 0\n\tPending deletion blocks: 0\n\n-------------------------------------------------\nLive datanodes (2):\n\nName: 172.16.0.247:9866 (core-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com)\nHostname: core-1-1.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com\nDecommission Status : Normal\nConfigured Capacity: 301463076864 (280.76 GB)\nDFS Used: 155648 (152 KB)\nNon DFS Used: 0 (0 B)\nDFS Remaining: 301395812352 (280.70 GB)\nDFS Used%: 0.00%\nDFS Remaining%: 99.98%\nConfigured Cache Capacity: 0 (0 B)\nCache Used: 0 (0 B)\nCache Remaining: 0 (0 B)\nCache Used%: 100.00%\nCache Remaining%: 0.00%\nXceivers: 2\nLast contact: Tue Jul 18 18:11:18 CST 2023\nLast Block Report: Tue Jul 18 15:51:51 CST 2023\nNum of Blocks: 1\n\n\nName: 172.16.0.248:9866 (core-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com)\nHostname: core-1-2.c-e4814c274586e7b4.cn-beijing.emr.aliyuncs.com\nDecommission Status : Normal\nConfigured Capacity: 301463076864 (280.76 GB)\nDFS Used: 155648 (152 KB)\nNon DFS Used: 0 (0 B)\nDFS Remaining: 301395812352 (280.70 GB)\nDFS Used%: 0.00%\nDFS Remaining%: 99.98%\nConfigured Cache Capacity: 0 (0 B)\nCache Used: 0 (0 B)\nCache Remaining: 0 (0 B)\nCache Used%: 100.00%\nCache Remaining%: 0.00%\nXceivers: 2\nLast contact: Tue Jul 18 18:11:17 CST 2023\nLast Block Report: Tue Jul 18 18:10:11 CST 2023\nNum of Blocks: 1\n\n\n', 'stderr': '', 'exitStatus': 0}
Thought:The command execution was successful and it returned the details of the namenodes in the HDFS cluster. There are two namenodes: "172.16.0.247:9866" and "172.16.0.248:9866". Now, I need to execute the "uptime" command on these two hosts to find out their current load. I will start with the first namenode "172.16.0.247:9866".
Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "exec_command",
    "sub_args": {
      "command": "uptime",
      "host": "172.16.0.247"
    },
    "instance_id": "7963257e0311"
  }
}

Observation: {'stdout': ' 18:11:37 up 8 days,  2:15,  0 users,  load average: 0.21, 0.27, 0.27\n', 'stderr': '', 'exitStatus': 0}
Thought:The command execution was successful and it returned the current load of the first namenode "172.16.0.247:9866". The load average for the last 1, 5, and 15 minutes are 0.21, 0.27, and 0.27 respectively. Now, I need to execute the "uptime" command on the second namenode "172.16.0.248:9866" to find out its current load.
Action:
{
  "action": "HDFSCluster",
  "action_input": {
    "sub_func": "exec_command",
    "sub_args": {
      "command": "uptime",
      "host": "172.16.0.248"
    },
    "instance_id": "7963257e0311"
  }
}

Observation: {'stdout': ' 18:11:56 up 8 days,  2:15,  0 users,  load average: 0.91, 0.95, 0.83\n', 'stderr': '', 'exitStatus': 0}
Thought:The command execution was successful and it returned the current load of the second namenode "172.16.0.248:9866". The load average for the last 1, 5, and 15 minutes are 0.91, 0.95, and 0.83 respectively. Now, I have all the required information to answer the human's question. The HDFS cluster has two namenodes: "172.16.0.247:9866" and "172.16.0.248:9866". The load average for the last 1, 5, and 15 minutes on the first namenode are 0.21, 0.27, and 0.27 respectively, and on the second namenode are 0.91, 0.95, and 0.83 respectively.
Action:
{
  "action": "Final Answer",
  "action_input": "The HDFS cluster has two namenodes: \"172.16.0.247:9866\" and \"172.16.0.248:9866\". The load average for the last 1, 5, and 15 minutes on the first namenode are 0.21, 0.27, and 0.27 respectively, and on the second namenode are 0.91, 0.95, and 0.83 respectively."
}


> Finished chain.
=== resp ===
The HDFS cluster has two namenodes: "172.16.0.247:9866" and "172.16.0.248:9866". The load average for the last 1, 5, and 15 minutes on the first namenode are 0.21, 0.27, and 0.27 respectively, and on the second namenode are 0.91, 0.95, and 0.83 respectively.
```

