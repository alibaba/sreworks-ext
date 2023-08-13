#!/usr/bin/python
# -*- coding: UTF-8 -*-

import langchain.tools.base
from langchain.tools import StructuredTool
from langchain.tools.base import create_schema_from_function
from langchain.agents import initialize_agent, AgentType
import inspect
import uuid
from typing import Dict
import json


class AI():
    _tools = {}
    _agents = {}

    def __init__(self):
        pass
        # print(self._tools)

    @staticmethod
    def _register_tool(obj, func, name=None, args_schema=None):
        if name is None:
            name = func.__name__
        obj._tools[name] = StructuredTool.from_function(func, name=name, args_schema=args_schema)

    @staticmethod
    def _register_agent(obj, name, agent):
        obj._agents[name] = agent
        pass

    @staticmethod
    def _make_class_tool_function(obj, class_tool):
        use_class = class_tool
        class_instances = {}

        def class_tool_template(sub_func: str, sub_args: Dict, instance_id: str = None) -> Dict:
            nonlocal use_class
            nonlocal class_instances
            if sub_func == "__init__":
                instance_id = str(uuid.uuid4()).split("-")[-1]
                class_instances[instance_id] = class_tool(**sub_args)
                return {"instance_id": instance_id}
            else:
                if instance_id is None:
                    return {"message": "not __init__ sub_func require instance_id parameter"}
                class_instance = class_instances[instance_id]
                if not hasattr(class_instance, sub_func):
                    return {"message": f"sub_func {sub_func} is not exist"}
                exec_func = getattr(class_instance, sub_func)
                return exec_func(**sub_args)

        class_tool_template.__name__ = class_tool.__name__
        class_tool_template.__doc__ = class_tool.__doc__
        class_tool_template.__doc__ += " The use of this tool is similar to a class, it requires initialization. You can use this tool by following these steps: First, call the '__init__' sub_func to instantiate, this call will return an 'instance_id'. Then, use sub_func=...,sub_args=...,instance_id=... to call the other functions on this instance.\n"
        class_tool_template.__doc__ += "    Below are the available sub_func list for this tool:\n"

        functions = [member for member in inspect.getmembers(class_tool, inspect.isfunction) if
                     ((not member[0].startswith('_')) or member[0] == "__init__")]
        for fn in functions:
            fn_properties = create_schema_from_function(f"Schema", fn[1]).schema()["properties"]
            fn_properties.pop('self', None)
            fn_properties_str = str(fn_properties).replace("{", "{{").replace("}", "}}")
            class_tool_template.__doc__ += f"    - sub_func: {fn[0]}  sub_args: " + fn_properties_str + "\n"

        class_tool_template.__doc__ += "\n"
        return class_tool_template

    @staticmethod
    def tools(obj, tools, agents):
        agent_tools = []

        for tool in tools:
            if tool in obj._tools:
                agent_tools.append(obj._tools[tool])
            elif inspect.isclass(tool):
                class_tool = obj._make_class_tool_function(obj, tool)
                a = create_schema_from_function(f"{class_tool.__name__}Schema", class_tool)
                agent_tools.append(StructuredTool.from_function(class_tool))
            elif callable(tool):
                agent_tools.append(StructuredTool.from_function(tool))
            else:
                class_funcs = [attr_name for attr_name in dir(tool) if not attr_name.startswith('_') and callable(getattr(tool, attr_name)) and getattr(tool, attr_name).__doc__]
                if len(class_funcs) > 0:
                    for fn in class_funcs:
                        agent_tools.append(StructuredTool.from_function(getattr(tool, fn)))
                else:
                    raise Exception("Tool " + tool + " not found")

        for ag in agents:
            if ag not in obj._tools:
                raise Exception("Agent " + ag + " not found")
            agent_tools.append(obj._tools[ag])

        return agent_tools

    def run(self, query, llm, tools=[], agents=[], verbose=False):
        agent_executor = initialize_agent(
            self.tools(self, tools, agents),
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        )
        return agent_executor.run(query)

    @staticmethod
    def _decorator_tool(obj):
        def decorator(func):
            obj._register_tool(func)

            def wrapper(*args, **kw):
                return func(*args, **kw)

            return wrapper
        return decorator

    def tool(self):
        return self._decorator_tool(self)

    @classmethod
    def init_tool(cls):
        return cls._decorator_tool(cls)

    @staticmethod
    def _decorator_agent(obj, llm, tools=[], agents=[], verbose=False):
        def decorator(func):
            obj._register_agent(obj, func.__name__, initialize_agent(
                obj.tools(obj, tools, agents),
                llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=verbose,
            ))

            def wrapper(*args, **kw):
                prompt = func(*args, **kw)
                return obj._agents[func.__name__].run(prompt)

            wrapper.__doc__ = func.__doc__
            args_schema = create_schema_from_function(f"{func.__name__}Schema", func)
            obj._register_tool(obj, wrapper, func.__name__, args_schema=args_schema)

            return wrapper

        return decorator


    def agent(self, llm, tools=[], agents=[], verbose=False):
        return self._decorator_agent(self, llm=llm, tools=tools, agents=agents, verbose=verbose)

    @classmethod
    def init_agent(cls, llm, tools=[], agents=[], verbose=False):
        return cls._decorator_agent(cls, llm=llm, tools=tools, agents=agents, verbose=verbose)
