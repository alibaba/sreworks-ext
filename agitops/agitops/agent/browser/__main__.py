#!/usr/bin/env python3

import os
import sys
import tempfile
import yaml
import json
import subprocess
import asyncio
import urllib.request

from openai import OpenAI
from ...util import run_command
from ...tool import ToolHandler
from browser_use import Controller, ActionResult, Agent, BrowserSession
from langchain_openai import ChatOpenAI


controller = Controller()

class ExplorerAgent():
    sys_prompt = "你来扮演一个任务专家，你是在一个linux系统(bookworm)中, 当前有planner已经对用户问题进行了任务拆分，请合理地使用工具解决 <task>..</task> 中提到的问题。"

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

    def record_event(self, content):
        self.event_path = os.path.join(self.conf["work_root_path"], "event")
        os.makedirs(self.event_path, exist_ok=True)

        print("record_event")
        print(content)
        h = open(self.event_path + "/exec.md", 'w')
        h.write(content)
        h.close()

    async def run(self):

        context_file = os.path.join(self.conf["task_path"], "context.json")

        agent = Agent(
            task=self.init_task(),
            enable_memory=False,
            llm=ChatOpenAI(
              model=self.conf["llm_model"],
              api_key=self.conf["llm_api_key"],
              base_url=self.conf["llm_api_url"],
            ),
            browser_session=BrowserSession(
                headless=True,
                viewport={'width': 1440, 'height': 900},
                args=['--no-sandbox'],
            ),
            override_system_message=self.sys_prompt,
            controller=controller,
        )
        result = await agent.run(max_steps=30)

        print(f"result: {result}")

        output = {}
        output["final_result"] = result.final_result()
        output["history"] = []

        for h in result.history:
            result = []
            for r in h.result:
                if r.extracted_content:
                    result.append({
                        "content": r.extracted_content,
                        "error": r.error,
                    })
            output["history"].append({
                "url": h.state.url,
                "result": result,
            })

        print(json.dumps(output, indent=4, ensure_ascii=False))

        h = open(context_file, 'w')
        h.write(json.dumps(output, indent=4, ensure_ascii=False))
        h.close()
   
        with open(os.path.join(self.conf["task_path"], "title.md"), "r") as h:
            taskTitle = h.read()

        h = open(os.path.join(self.conf["task_path"], "output.md"), "w")
        h.write(output["final_result"])
        h.close()
        self.record_event("### " + taskTitle + "\n" + output["final_result"])

        self.git_commit(output["final_result"])


if __name__ == "__main__":

    agent = ExplorerAgent("/etc/gitops.yaml")

    @controller.action('bash')
    def exec_shell(exec_command: str) -> ActionResult:
        process = subprocess.Popen(
            exec_command,
            shell=True,
            cwd=agent.workspace_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(f"exec_command: {exec_command}")
        stdout, stderr = process.communicate()
        return_code = process.returncode
        result = f"ret:{return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
        print(result)
        return ActionResult(extracted_content=result)

    @controller.action('google')
    def search(query: str) -> ActionResult:
        google_api_key = agent.conf.get("google_api_key")
        google_api_cx = agent.conf.get("google_api_cx")

        url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_api_cx}&q={query}"

        result = ""
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    print(json.dumps(data, indent=4, ensure_ascii=False))
                else:
                    result = f"请求失败，状态码: {response.status}"
                    print(result)
        except urllib.error.URLError as e:
            result = f"请求出错: {e.reason}"
            print(result)

        return ActionResult(extracted_content=result)

    asyncio.run(agent.run())
