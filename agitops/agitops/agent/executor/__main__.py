#!/usr/bin/env python3

import os
import sys
import tempfile
import yaml
import json
from openai import OpenAI
from ...util import run_command
from ...tool import ToolHandler

# sys_prompt = "你来扮演一个任务专家，你是在一个linux系统(alpine)中, 当前有planner已经对用户问题进行了任务拆分，请合理地使用工具解决 <task>..</task> 中提到的问题。"

class ExecutorAgent():
    sys_prompt = "你来扮演一个任务专家，你是在一个linux系统(alpine)中, 当前有planner已经对用户问题进行了任务拆分，请合理地使用工具解决 <task>..</task> 中提到的问题。"

    def __init__(self, conf_path):
        # /gitops.yaml
        # llm_model: qwen-max
        # llm_api_url: http://
        # llm_api_key: *****
        # work_root_path: *
        # task_name: *

        if os.path.exists(conf_path):
            h = open(conf_path, "r")
            conf_yaml = h.read()
            h.close()
            print(f"[{conf_path}]")
            print(conf_yaml)
            self.conf = yaml.safe_load(conf_yaml)
        else:
            print(f"not found {conf_path}")
            sys.exit(1)

        conf_keys = ["llm_model", "llm_api_url", "llm_api_key", "work_root_path", "task_path", "task_name",
                     "git_user_email", "git_user_name"]
        for conf_key in conf_keys:
            if conf_key not in self.conf:
                print(f"not found {conf_key} in {conf_path}")
                sys.exit(1)

        self.workspace_path = os.path.join(self.conf["work_root_path"], "workspace")

        if os.path.isfile(self.workspace_path):
            os.remove(self.workspace_path)
            os.makedirs(self.workspace_path, exist_ok=True)
            # 存放一个文件，避免第一轮没有文件
            with open(os.path.join(self.workspace_path, ".gitignore"), 'w') as f:
                pass 

        self.toolHandler = ToolHandler({"bash": "bash.py"})

        self.llm_client = OpenAI(
            api_key=self.conf["llm_api_key"],
            base_url=self.conf["llm_api_url"],
        )

    def init_task(self):
        terminalOutput = ""
        (ret, stdout, stderr) = run_command(["pwd"], self.workspace_path)
        terminalOutput += "> pwd\n"
        terminalOutput += stdout.strip() + "\n"

        (ret, stdout, stderr) = run_command(["tree"], self.workspace_path)
        terminalOutput += "> tree\n"
        terminalOutput += stdout.strip() + "\n"

        with open(os.path.join(self.conf["task_path"], "input.md"), "r") as h:
            taskInput = h.read()

        with open(os.path.join(self.conf["task_path"], "user_message.md"), "r") as h:
            userMessage = h.read()

        terminalOutput += f"用户的问题是: {userMessage} \n"
        terminalOutput += f"planner拆分给任务专家的任务是: \n"
        terminalOutput += "<task>\n" + taskInput + "</task>\n"
        return terminalOutput

    def git_commit(self, message):
        git_conf_commands = ["git",
                    '-c', f"user.email={self.conf["git_user_email"]}",
                    '-c', f"user.name={self.conf["git_user_name"]}",
                    '-c', f"safe.directory={self.conf['work_root_path']}"]

        (ret, stdout, stderr) = run_command(git_conf_commands + ["add", "."], cwd=self.conf["work_root_path"])
        if ret != 0:
            print(f"git add failed: stdout:{stdout} stderr:{stderr}", flush=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(f"[{self.conf['task_name']}] " + message)

        commands = git_conf_commands + ["commit", "-F", f.name]
        print(commands)

        (ret, stdout, stderr) = run_command(commands, cwd=self.conf["work_root_path"])
        if ret != 0:
            print(f"git commit failed: stdout:{stdout} stderr:{stderr}", flush=True)

    def run(self):

        context_file = os.path.join(self.conf["task_path"], "context.json")
        if os.path.exists(context_file):
            h = open(context_file, 'r')
            messages = json.loads(h.read())
            h.close()
        else:
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": self.init_task()}]

        completion = self.llm_client.chat.completions.create(
            model=self.conf["llm_model"],
            messages=messages,
            tools=self.toolHandler.get_schema()
        ).to_dict()

        waitFunction = False
        messageData = None

        if completion.get("choices") is not None:
            messageData = completion["choices"][0].get("message")

        if messageData is None:
            raise Exception("no message")

        messages.append(messageData)
        lastMsgContent = messageData.get("content")
        print(lastMsgContent)
        if messageData.get("tool_calls") is not None:
            if lastMsgContent in [None, ""]:
                lastMsgContent = "call tool"
            for tool_call in messageData["tool_calls"]:
                waitFunction = True
                print(f"exec tool {tool_call}")
                message = self.toolHandler.exec_tool(
                    name=tool_call.get("function", {}).get("name"),
                    arguments=tool_call.get("function", {}).get("arguments", "{}"),
                    tool_call_id=tool_call["id"],
                    cwd=self.workspace_path,
                )
                print(f"exec tool result {message}")
                messages.append(message)

        if waitFunction is False:
            h = open(os.path.join(self.conf["task_path"], "output.md"), "w")
            h.write(lastMsgContent)
            h.close()

        h = open(context_file, 'w')
        h.write(json.dumps(messages, indent=4, ensure_ascii=False))
        h.close()

        self.git_commit(lastMsgContent)
        return waitFunction


if __name__ == "__main__":

    agent = ExecutorAgent("/etc/gitops.yaml")

    next_loop = True
    count = 0
    while next_loop:
        if count > 30:
            print("max loop 30")
            sys.exit(1)
        count += 1
        print(f"\nthought loop: {count}", flush=True)
        next_loop = agent.run()
