# Linux运行时长

## 使用工具
- [ssh](../../tools/Linux/SSH/)

## 使用方法
```
resp = ai.run("帮我看看 8.130.35.2 这台机器启动了多久了", llm=OpenAI(temperature=0.5, model_name="gpt-4"), tools=[ssh], verbose=True)
```

## 调用过程

```
> Entering new AgentExecutor chain...
The user is asking for the uptime of the server at IP address 8.130.x.x. I can use the ssh tool to connect to the server and run the 'uptime' command to get this information.

Action:
{
  "action": "ssh",
  "action_input": {
    "command": "uptime",
    "host": "8.130.x.x"
  }
}
Observation: {'stdout': ' 14:03:59 up 22 days, 22:56,  0 users,  load average: 1.17, 1.78, 2.28\n', 'stderr': '', 'exitStatus': 0}
Thought:The ssh tool has returned the output of the 'uptime' command. The server has been up for 22 days, 22 hours, and 56 minutes. I will relay this information to the user.

Action:
{
  "action": "Final Answer",
  "action_input": "这台服务器已经运行了22天22小时56分钟。"
}

> Finished chain.
=== resp ===
这台服务器已经运行了22天22小时56分钟。


```


