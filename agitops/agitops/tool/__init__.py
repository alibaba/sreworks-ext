#!/usr/bin/env python3

import argparse as _argparse
import json
import sys
from ..util import run_command
import os
import tempfile


class ArgumentParserProxy:
    def __init__(self, *args, **kwargs):
        self._parser = _argparse.ArgumentParser(*args, **kwargs)
        self._parser.add_argument("--json-input", help="input by json file", required=False)
        self._subparsers = []

    def add_subparsers(self, **kwargs):
        subparsers = SubParsersActionProxy(self._parser.add_subparsers(**kwargs))
        self._subparsers.append(subparsers)
        return subparsers

    def parse_args(self, *args, **kwargs):
        args = self._parser.parse_args(*args, **kwargs)

        if args.json_input:
            json_input = json.load(open(args.json_input, "r"))

            # 未传入的默认值补全
            if len(self._subparsers) > 0 and self._subparsers[0]._subparsers.dest in json_input:
                choice = json_input[self._subparsers[0]._subparsers.dest]
                for action in self._subparsers[0]._subparsers.choices[choice]._actions:
                    if action.required is False and action.dest not in json_input:
                        json_input[action.dest] = None

            args = _argparse.Namespace(**json_input)

        return args

    def __getattr__(self, name):
        return getattr(self._parser, name)


class SubParsersActionProxy:
    def __init__(self, subparsers):
        self._subparsers = subparsers

    def add_parser(self, *args, **kwargs):
        return self._subparsers.add_parser(*args, **kwargs)

    def dump_schema(self):
        functions = []
        for action in self._subparsers._get_subactions():
            params = {}
            requiredList = []
            for param in self._subparsers.choices[action.dest]._actions:
                if param.dest == "help":
                    continue
                params[param.dest] = {
                    "description": param.help,
                    "type": "string"
                }
                if param.required:
                    requiredList.append(param.dest)

            fn = {
                "type": "function",
                "function": {
                    "name": action.dest,
                    "description": action.help,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": requiredList
                    }
                }
            }
            functions.append(fn)
        print(json.dumps(functions, indent=4))

    def __getattr__(self, name):
        return getattr(self._subparsers, name)


class argparse():
    @staticmethod
    def ArgumentParser(*args, **kwargs):
        return ArgumentParserProxy(*args, **kwargs)


class ToolHandler():
    tools = {}

    def __init__(self, path_map):
        self_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        for tool_name, path in path_map.items():
            if "#" in tool_name:
                raise Exception(f"tool_name {tool_name} exist #")
            (ret, stdout, stderr) = run_command(f"{sys.executable} {path}", self_path)
            if ret != 0:
                raise RuntimeError(f"Failed to execute {path}: {stderr}")
            if "/" not in path:
                path = self_path + "/" + path

            schema = json.loads(stdout.strip())
            for tool in schema:
                tool["function"]["name"] = f"{tool_name}#{tool['function']['name']}"
            self.tools[tool_name] = {
                "path": path,
                "schema": schema
            }

    def get_schema(self):
        tools = []
        for tool_name, tool_info in self.tools.items():
            tools += tool_info["schema"]
        return tools

    def exec_tool(self, name, arguments, tool_call_id, cwd, envs={}):
        # {"role": "tool", "content": "工具的输出", "tool_call_id": completion.choices[0].message.tool_calls[0].id}
        result = {
            "role": "tool",
            "tool_call_id": tool_call_id,
        }
        tool_name, function_name = name.split("#", 1)
        tool_object = self.tools[tool_name]
        print(tool_object)
        match_functions = [tool for tool in tool_object["schema"] if tool["function"]["name"] == name]
        if len(match_functions) == 0:
            result["content"] = f"tool {name} not found"
            return result

        try:
            args = json.loads(arguments)
        except Exception as e:
            result["content"] = f"parse arguments json failed"
            return result

        # 创建一个临时文件并写入 arguments 内容
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            args["command"] = function_name
            f.write(json.dumps(args))

        (ret, stdout, stderr) = run_command(f"{sys.executable} {tool_object['path']} --json-input {f.name}", cwd=cwd, envs=envs)
        result["content"] = f"ret:{ret}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
        return result
