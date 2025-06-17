#!/usr/bin/env python3

import os
import sys
import yaml
import json
import time
from ...util import run_command
from openai import OpenAI
import tempfile

self_path = os.path.abspath(os.path.dirname(__file__))


def user_query(user_message):
    queryOutput = "<question>\n" + user_message["message"] + "</question>\n"
    if len(user_message["files"]) > 0:
        queryOutput += "<questionFiles>" + json.dumps(user_message["files"]) + "</questionFiles>\n"
    return queryOutput


def parse_yaml(plan_yaml):
    if "```yaml" in plan_yaml:
        raw = plan_yaml.split("```yaml")[1].split("```")[0]
    elif "```" in plan_yaml:
        raw = plan_yaml.split("```")[1].split("```")[0]
    else:
        raw = plan_yaml
    return yaml.safe_load(raw)

def set_plan_id(plan, parent_task_id=None):
    count = 0
    for task in plan:
        count += 1
        if parent_task_id is not None:
            task["id"] = f"{parent_task_id}-{count}"
        else:
            task["id"] = f"task{count}"
        if task.get("children") is not None and len(task["children"]) > 0:
            set_plan_id(task["children"], parent_task_id=task["id"])
    return plan


def find_todo_task(plan):
    for task in plan:
        if task.get("finish") is not True:
            if task.get("children") is not None and len(task["children"]) > 0:
                children_todo_task = find_todo_task(task["children"])
                return children_todo_task if children_todo_task is not None else task
            else:
                return task


def set_task_result(plan, task_id, result):
    for task in plan:
        if task["id"] == task_id:
            task["result"] = result
            task["finish"] = True
            return True
        elif task_id.startswith(task["id"]):
            return set_task_result(task["children"], task_id, result)
    return False

def plan_to_markdown(plan, indent=0):
    md = ""
    for task in plan:
        if "description" in task:
            description = f"`{' '.join(task['description'].split())}`"
        else:
            description = ""
        md += " " * indent + f"- {task['title']} {description}\n"
        if task.get("children") is not None and len(task["children"]) > 0:
            md += plan_to_markdown(task["children"], indent+2)
    return md


class PlanAgent():
    planPrompt = """
你是一个任务管理专家，你不用直接执行任务，请合理地提出计划解决 <question>..</question> 中提到的问题。
<questionFiles> .. </questionFiles>为用户上传的与问题相关的文件。
整个计划任务请使用YAML数组编写，简单任务步骤最好不要超过6步，没有子任务。
每个任务名使用title字段，任务描述使用description字段，每个任务的description需要是一个明确要解决的问题，而不是答案。
每个任务的executor字段可以指定是browser或explorer。browser是在浏览器中完成任务，explorer是在Linux文件系统中完成任务，默认为explorer。
"""

#     reviewPrompt = """
# 你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
# <plan>...</plan> 是另外一个专家设计的任务计划。请从下面这些点进行评估优化。
# 1. 每个任务的嵌套子任务不能只有一个，如果有这种情况则将其与父任务合并一个。
# 2. 分解的任务最终不要包含去其他平台发帖寻求帮助等无法在确定时间获得反馈的任务。
# 3. 请根据用户问题难度分析当前的任务计划拆分是否合适，如果不合适则请进行调整。
# 4. 避免出现顶层只有一个父任务的情况，如果出现，请将顶层的任务取消，下面子任务往上提。
# 请估计上面这些点，优化<plan></plan>中的任务计划，再次给出YAML。
# """

    finalPrompt = """
你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
<result>...</result> 是当前执行器根据任务计划的执行返回，请最终总结本次任务。
"""

    taskPrompt = """
你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
<task>...</task> 是当前需要分配给这个执行器的任务。
请整理整合任务内容，使得执行器能够只根据任务内容来执行任务。
请确保输出的任务内容中包含了 <question> <questionFiles> 部分与<task>相关的必要信息。
"""


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

        conf_keys = ["llm_model", "llm_api_url", "llm_api_key", "work_root_path", "preview_plan_markdown",
                     "git_user_email", "git_user_name"]
        for conf_key in conf_keys:
            if conf_key not in self.conf:
                print(f"not found {conf_key} in {conf_path}")
                sys.exit(1)

        self.chat_path = os.path.join(self.conf["work_root_path"], "chat")
        self.task_path = os.path.join(self.conf["work_root_path"], "task")


        self.llm_client = OpenAI(
            api_key=self.conf["llm_api_key"],
            base_url=self.conf["llm_api_url"],
        )

    def call_llm(self, sysPrompt, userPrompt):
        messages = [
            {"role": "system", "content": sysPrompt},
            {"role": "user", "content": userPrompt}]
        completion = self.llm_client.chat.completions.create(
            model=self.conf["llm_model"],
            messages=messages,
        ).to_dict()
        return completion["choices"][0]["message"]["content"]

    def load_chats(self):
        chats = []
        for message_file in os.listdir(self.chat_path):
            (role, message_time) = message_file.split("_")
            message_data = {
                "role": role,
                "time": message_time,
                "message": None,
                "files": [],
                "path": os.path.join(self.chat_path, message_file)
            }
            files = os.listdir(os.path.join(self.chat_path, message_file))
            if "message.md" in files:
                h = open(self.chat_path + "/" + message_file + "/message.md", "r")
                message_data["message"] = h.read()
                h.close()
                files.remove("message.md")
            if "plan.json" in files:
                h = open(self.chat_path + "/" + message_file + "/plan.json", "r")
                message_data["plan"] = json.loads(h.read())
                h.close()
                files.remove("plan.json")
            message_data["files"] = files
            chats.append(message_data)

        # 按照time字段倒序排序
        chats = sorted(chats, key=lambda x: x["time"], reverse=True)
        return chats

    def init_assistant_message(self):
        current_time = int(time.time())
        directory_name = f"assistant_{current_time}"
        directory_path = os.path.join(self.chat_path, directory_name)
        os.makedirs(directory_path, exist_ok=True)

        return {
            "role": "assistant",
            "time": current_time,
            "message": None,
            "plan": [],
            "path": directory_path,
        }

    def record_event(self, content):
        self.event_path = os.path.join(self.conf["work_root_path"], "event")
        os.makedirs(self.event_path, exist_ok=True)

        print("record_event")
        print(content)
        h = open(self.event_path + "/plan.md", 'w')
        h.write(content)
        h.close()


    def checkout_latest_message(self, chats):
        if len(chats) == 0:
            raise ValueError("No chat history found.")

        if chats[0]["role"] != "assistant":
            return self.init_assistant_message(), chats[0]
        elif chats[0]["role"] == "assistant":
            return chats[0], chats[1]
        else:
            raise ValueError("Unexpected chat history format.")

    def git_commit(self, message):
        git_conf_commands = ["git",
                    '-c', f"user.email={self.conf["git_user_email"]}",
                    '-c', f"user.name={self.conf["git_user_name"]}",
                    '-c', f"safe.directory={self.conf['work_root_path']}"]

        (ret, stdout, stderr) = run_command(git_conf_commands + ["add", "."], cwd=self.conf["work_root_path"])
        if ret != 0:
            print(f"git add failed: stdout:{stdout} stderr:{stderr}", flush=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(message)

        commands = git_conf_commands + ["commit", "-F", f.name]
        print(commands)

        (ret, stdout, stderr) = run_command(commands, cwd=self.conf["work_root_path"])
        if ret != 0:
            print(f"git commit failed: stdout:{stdout} stderr:{stderr}", flush=True)

    def run(self):
        doing_task_flag_file = os.path.join(self.task_path, "doing-task")
        chats = self.load_chats()
        assistant_message, user_message = self.checkout_latest_message(chats)
        print(user_message, flush=True)

        if assistant_message["message"] is not None:
            print("final answer:")
            print(assistant_message["message"], flush=True)
            return

        if os.path.exists(assistant_message["path"] + "/plan.json"):
            h = open(assistant_message["path"] + "/plan.json", 'r')
            plan = json.loads(h.read())
            h.close()

            if os.path.exists(doing_task_flag_file):
                h = open(doing_task_flag_file, 'r')
                doing_role, doing_task_id = h.read().strip().split("/")
                h.close()
                os.remove(doing_task_flag_file)
                h = open(os.path.join(self.task_path, doing_task_id, "output.md"), 'r')
                doing_task_result = h.read().strip()
                h.close()
                set_task_result(plan, doing_task_id, doing_task_result)

        else:
            plan_yaml = self.call_llm(self.planPrompt, user_query(user_message))
            print("raw plan yaml :")
            print(plan_yaml, flush=True)
            # plan_yaml = self.call_llm(self.reviewPrompt, plan_yaml)
            # print("plan yaml after review :")
            # print(plan_yaml, flush=True)
            plan = parse_yaml(plan_yaml)
            plan = set_plan_id(plan)
            self.record_event("# 计划\n" + plan_to_markdown(plan))

        h = open(assistant_message["path"] + "/plan.json", 'w')
        h.write(json.dumps(plan, indent=4, ensure_ascii=False))
        h.close()

        h = open(self.conf["preview_plan_markdown"], 'w')
        h.write("```\n" + json.dumps(plan, indent=4, ensure_ascii=False) + "\n```\n")
        h.close()

        print(json.dumps(plan, indent=4, ensure_ascii=False), flush=True)

        todo_task = find_todo_task(plan)

        if todo_task is None:
            finalAnswer = self.call_llm(self.finalPrompt, user_query(user_message))
            
            h = open(assistant_message["path"] + "/message.md", 'w')
            h.write(finalAnswer)
            h.close()

            h = open(self.conf["preview_plan_markdown"], 'w')
            h.write(finalAnswer)
            h.close()

            self.record_event("# 结论\n" + finalAnswer)


        else:

            todo_task_path = self.task_path + "/" + todo_task["id"]
            if os.path.isfile(self.task_path):
                os.remove(self.task_path)
            if not os.path.exists(todo_task_path):
                os.makedirs(todo_task_path, exist_ok=True)

            task_input = self.call_llm(self.taskPrompt, f"<task>{todo_task}</task>")
            h = open(todo_task_path + "/input.md", 'w')
            h.write(task_input)
            h.close()

            h = open(todo_task_path + "/title.md", 'w')
            h.write(todo_task["title"])
            h.close()

            h = open(todo_task_path + "/user_message.md", 'w')
            h.write(user_message["message"])
            h.close()

            print(f"{todo_task['id']} start", flush=True)
            print(task_input, flush=True)
            h = open(doing_task_flag_file, 'w')
            h.write(f"{todo_task.get("executor","explorer")}/{todo_task['id']}")
            h.close()

        self.git_commit("Plan task")


if __name__ == "__main__":

    conf_path = "/etc/gitops.yaml"

    agent = PlanAgent("/etc/gitops.yaml")

    agent.run()
