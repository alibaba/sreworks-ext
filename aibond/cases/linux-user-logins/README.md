# Linux用户登录记录分析

## 使用工具
- [ssh](../../tools/Linux/SSH/)
- [ipinfo](../../tools/Linux/Ipinfo/)

## 使用方法
```
resp = ai.run("帮我分析 8.130.x.x 这台机器最近被来自哪些地方的IP登录过", llm=OpenAI(temperature=0.5, model_name="gpt-4"), tools=[ssh,ipinfo], verbose=True)
```

## 调用过程
```

> Entering new AgentExecutor chain...
The user is asking for an analysis of the recent login IP addresses for the machine with the IP 8.130.x.x. The first step is to use the ssh tool to connect to this machine and retrieve the login history.

Action:
{
  "action": "ssh",
  "action_input": {
    "command": "last -i",
    "host": "8.130.x.x"
  }
}

Observation: {'stdout': 'root     pts/0        140.205.x.x    Thu Jul  6 16:33 - 19:13  (02:40)    \nroot     pts/1        140.205.x.x    Thu Jul  6 09:54 - 13:08  (03:14)    \nroot     pts/0        140.205.x.x    Thu Jul  6 09:53 - 12:07  (02:13)    \nroot     pts/1        220.191.x.x  Wed Jul  5 22:07 - 00:47  (02:40)    \nroot     pts/0        220.191.x.x  Wed Jul  5 21:56 - 00:14  (02:17)    \nroot     pts/0        140.205.x.x     Wed Jul  5 18:24 - 20:38  (02:13)    \nroot     pts/0        140.205.x.x     Wed Jul  5 18:23 - 18:24  (00:01)    \nroot     pts/1        140.205.x.x     Wed Jul  5 12:02 - 14:17  (02:14)    \nroot     pts/0        140.205.x.x     Wed Jul  5 11:45 - 14:13  (02:28)    \nroot     pts/0        220.191.x.x  Wed Jul  5 06:40 - 06:40  (00:00)    \nroot     pts/0        140.205.x.x    Fri Jun 30 15:55 - 18:08  (02:13)    \nroot     pts/1        140.205.x.x     Mon Jun 19 13:38 - 16:04  (02:26)    \nroot     pts/0        140.205.x.x     Mon Jun 19 11:33 - 13:48  (02:14)    \nroot     pts/0        183.157.x.x   Sat Jun 17 06:30 - 10:20  (03:50)    \nroot     pts/0        140.205.x.x    Fri Jun 16 10:55 - 13:10  (02:15)    \nroot     pts/1        140.205.x.x     Wed Jun 14 16:09 - 18:40  (02:31)    \nroot     pts/0        140.205.x.x     Wed Jun 14 15:13 - 16:23  (01:09)    \nreboot   system boot  0.0.0.0          Wed Jun 14 15:07 - 13:43 (22+22:36)  \nroot     pts/0        140.205.x.x     Wed Jun 14 15:00 - down   (00:05)    \nreboot   system boot  0.0.0.0          Wed Jun 14 15:00 - 15:05  (00:05)    \nroot     pts/2        140.205.x.x     Wed Jun 14 14:45 - down   (00:10)    \nroot     pts/1        140.205.x.x     Wed Jun 14 14:44 - down   (00:11)    \nroot     pts/0        140.205.x.x     Wed Jun 14 14:14 - down   (00:41)    \nreboot   system boot  0.0.0.0          Wed Jun 14 14:13 - 14:55  (00:41)    \n\nwtmp begins Mon May 22 10:16:27 2023\n', 'stderr': '', 'exitStatus': 0}
Thought:The ssh command returned a list of recent logins with their corresponding IP addresses. The next step is to use the ipinfo tool to get the geographical location of each IP address. The IP addresses are: 140.205.x.x, 220.191.x.x, 140.205.x.x, 140.205.x.x, 140.205.x.x, 183.157.x.x, 140.205.x.x. I'll start with the first one: 140.205.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "140.205.x.x"
  }
}

Observation: {'ip': '140.205.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS37963 Hangzhou Alibaba Advertising Co.,Ltd.', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 140.205.x.x is located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the next IP address: 220.191.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "220.191.x.x"
  }
}

Observation: {'ip': '220.191.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS4134 CHINANET-BACKBONE', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 220.191.x.x is also located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the next IP address: 140.205.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "140.205.x.x"
  }
}

Observation: {'ip': '140.205.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS37963 Hangzhou Alibaba Advertising Co.,Ltd.', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 140.205.x.x is also located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the next IP address: 140.205.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "140.205.x.x"
  }
}

Observation: {'ip': '140.205.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS37963 Hangzhou Alibaba Advertising Co.,Ltd.', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 140.205.x.x is also located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the next IP address: 140.205.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "140.205.x.x"
  }
}

Observation: {'ip': '140.205.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS37963 Hangzhou Alibaba Advertising Co.,Ltd.', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 140.205.x.x is also located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the next IP address: 183.157.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "183.157.x.x"
  }
}

Observation: {'ip': '183.157.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS4134 CHINANET-BACKBONE', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 183.157.x.x is also located in Hangzhou, Zhejiang, China. Now I'll get the geographical location for the last IP address: 140.205.x.x.

Action:
{
  "action": "ipinfo",
  "action_input": {
    "ip": "140.205.x.x"
  }
}

Observation: {'ip': '140.205.x.x', 'city': 'Hangzhou', 'region': 'Zhejiang', 'country': 'CN', 'loc': '30.2936,120.1614', 'org': 'AS37963 Hangzhou Alibaba Advertising Co.,Ltd.', 'timezone': 'Asia/Shanghai', 'readme': 'https://ipinfo.io/missingauth'}
Thought:The IP address 140.205.x.x is also located in Hangzhou, Zhejiang, China. All the IP addresses that recently logged into the machine with the IP 8.130.x.x are located in Hangzhou, Zhejiang, China.

Action:
{
  "action": "Final Answer",
  "action_input": "最近登录到IP为8.130.x.x的机器的IP地址都位于中国浙江杭州。"
}


> Finished chain.
=== resp ===
最近登录到IP为8.130.x.x的机器的IP地址都位于中国浙江杭州。
```

