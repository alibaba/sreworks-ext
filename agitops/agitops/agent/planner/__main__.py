#!/usr/bin/env python3

import os
import sys
import yaml
import json
import subprocess
from ...util import run_command
from ...tool import ToolHandler
from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo
# from tzlocal import get_localzone
import tempfile

self_path = os.path.abspath(os.path.dirname(__file__))

planPrompt = """
你是一个任务管理专家，你不用直接执行任务，请合理地提出计划解决 <question>..</question> 中提到的问题。
<questionFiles> .. </questionFiles>为用户上传的与问题相关的文件。
整个计划任务请使用YAML数组编写，最顶层步骤最好不要超过6步。
如果有子任务直接使用YAML嵌套，子任务使用children作为key，每个任务名使用title，任务描述使用description。
每个任务的description需要是一个明确要解决的问题，而不是答案。
"""

reviewPrompt = """
你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
<plan>...</plan> 是另外一个专家设计的任务计划。请从下面这些点进行评估优化。
1. 每个任务的嵌套子任务不能只有一个，如果有这种情况则将其与父任务合并一个。
2. 分解的任务最终不要包含去其他平台发帖寻求帮助等无法在确定时间获得反馈的任务。
3. 请根据用户问题难度分析当前的任务计划拆分是否合适，如果不合适则请进行调整。
4. 避免出现顶层只有一个父任务的情况，如果出现，请将顶层的任务取消，下面子任务往上提。
请估计上面这些点，优化<plan></plan>中的任务计划，再次给出YAML。
"""

taskPrompt = """
你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
<task>...</task> 是当前需要分配给这个执行器的任务。
请整理整合任务内容，使得执行器能够只根据任务内容来执行任务。
请确保输出的任务内容中包含了 <question> <questionFiles> 部分与<task>相关的必要信息。
"""

finalPrompt = """
你是一个任务管理专家，当前用户的问题是 <question>..</question> <questionFiles> .. </questionFiles>中。
<result>...</result> 是当前执行器根据任务计划的执行返回，请最终总结本次任务。
"""


def call_llm(llm_client, llm_model, systemPrompt, userPrompt):
    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt}]
    completion = llm_client.chat.completions.create(
        model=llm_model,
        messages=messages,
    ).to_dict()
    return completion["choices"][0]["message"]["content"]


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


def execute_agent(work_root_path, llm_model, llm_client, preview_plan_markdown):
    chat_path = work_root_path + "/chat"
    task_path = work_root_path + "/task"
    doing_task_flag_file = task_path + "/doing-task"
    chats = load_chats(chat_path)
    assistant_message, user_message = checkout_latest_message(chats, chat_path)

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
            h = open(os.path.join(task_path, doing_task_id, "output.md"), 'r')
            doing_task_result = h.read().strip()
            h.close()
            set_task_result(plan, doing_task_id, doing_task_result)

    else:
        plan_yaml = call_llm(llm_client, llm_model, planPrompt, user_query(user_message))
        print("raw plan yaml :")
        print(plan_yaml, flush=True)
        plan_yaml = call_llm(llm_client, llm_model, reviewPrompt, plan_yaml)
        print("plan yaml after review :")
        print(plan_yaml, flush=True)
        plan = parse_yaml(plan_yaml)
        plan = set_plan_id(plan)

    h = open(assistant_message["path"] + "/plan.json", 'w')
    h.write(json.dumps(plan, indent=4, ensure_ascii=False))
    h.close()

    h = open(preview_plan_markdown, 'w')
    h.write("```\n" + json.dumps(plan, indent=4, ensure_ascii=False) + "```\n")
    h.close()

    print(json.dumps(plan, indent=4, ensure_ascii=False), flush=True)

    todo_task = find_todo_task(plan)
    if todo_task is None:
        finalAnswer = call_llm(llm_client, llm_model, finalPrompt, user_query(user_message))
        
        h = open(assistant_message["path"] + "/message.md", 'w')
        h.write(finalAnswer)
        h.close()

        h = open(preview_plan_markdown, 'w')
        h.write(finalAnswer)
        h.close()

        return

    todo_task_path = task_path + "/" + todo_task["id"]
    if os.path.isfile(task_path):
        os.remove(task_path)
    if not os.path.exists(todo_task_path):
        os.makedirs(todo_task_path, exist_ok=True)

    task_input = call_llm(llm_client, llm_model, taskPrompt, f"<task>{todo_task}</task>")
    h = open(todo_task_path + "/input.md", 'w')
    h.write(task_input)
    h.close()

    h = open(todo_task_path + "/user_message.md", 'w')
    h.write(user_message["message"])
    h.close()

    print(f"{todo_task['id']} start", flush=True)
    print(task_input, flush=True)
    h = open(doing_task_flag_file, 'w')
    h.write(f"executor/{todo_task['id']}")
    h.close()


def load_chats(chat_path):
    chats = []
    for message_file in os.listdir(chat_path):
        (role, message_time) = message_file.split("_")
        message_data = {
            "role": role,
            "time": message_time,
            "message": None,
            "files": [],
            "path": os.path.join(chat_path, message_file)
        }
        files = os.listdir(chat_path + "/" + message_file)
        if "message.md" in files:
            h = open(chat_path + "/" + message_file + "/message.md", "r")
            message_data["message"] = h.read()
            h.close()
            files.remove("message.md")
        if "plan.json" in files:
            h = open(chat_path + "/" + message_file + "/plan.json", "r")
            message_data["plan"] = json.loads(h.read())
            h.close()
            files.remove("plan.json")
        message_data["files"] = files
        chats.append(message_data)

    # 按照time字段倒序排序
    chats = sorted(chats, key=lambda x: x["time"], reverse=True)
    return chats


def init_assistant_message(chat_path):
    current_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M%S")
    # current_time = get_current_time()
    directory_name = f"assistant_{current_time}"
    directory_path = os.path.join(chat_path, directory_name)
    os.makedirs(directory_path, exist_ok=True)

    return {
        "role": "assistant",
        "time": current_time,
        "message": None,
        "plan": [],
        "path": directory_path,
    }


def checkout_latest_message(chats, chat_path):
    if len(chats) == 0:
        raise ValueError("No chat history found.")

    if chats[0]["role"] != "assistant":
        return init_assistant_message(chat_path), chats[0]
    elif chats[0]["role"] == "assistant":
        return chats[0], chats[1]


if __name__ == "__main__":

    conf_path = "/etc/gitops.yaml"

    if os.path.exists(conf_path):
        h = open(conf_path, "r")
        conf_yaml = h.read()
        h.close()
        print(f"[{conf_path}]")
        print(conf_yaml)
        conf = yaml.safe_load(conf_yaml)
    else:
        print(f"not found {conf_path}")
        sys.exit(1)

    conf_keys = ["llm_model", "llm_api_url", "llm_api_key", "work_root_path","preview_plan_markdown"]
    for conf_key in conf_keys:
        if conf_key not in conf:
            print(f"not found {conf_key} in {conf_path}")
            sys.exit(1)

    execute_agent(conf["work_root_path"], conf["llm_model"], llm_client=OpenAI(
        api_key=conf["llm_api_key"],
        base_url=conf["llm_api_url"],
    ), preview_plan_markdown=conf["preview_plan_markdown"])
