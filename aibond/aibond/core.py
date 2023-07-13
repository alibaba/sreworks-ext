#!/usr/bin/python
# -*- coding: UTF-8 -*-

from langchain.tools import StructuredTool
from langchain.tools.base import create_schema_from_function
from langchain.agents import initialize_agent, AgentType

class AI():
  _tools = {}
  _agents = {}

  def _register_tool(self, func, name=None, args_schema=None):
      if name is None:
          name = func.__name__
      self._tools[name] = StructuredTool.from_function(func, name=name, args_schema=args_schema)

  def _register_agent(self, name, agent):
      self._agents[name] = agent
      pass

  def tools(self, tools, agents):
      agent_tools = []

      for tool in tools:
          if tool in self._tools:
              agent_tools.append(self._tools[tool])
          elif callable(tool):
              agent_tools.append(StructuredTool.from_function(tool))
          else:
              raise Exception("Tool " + tool + " not found")

      for ag in agents:
          if ag not in self._tools:
              raise Exception("Agent " + ag + " not found")
          agent_tools.append(self._tools[ag])

      return agent_tools

  def run(self, query, llm, tools=[], agents=[], verbose=False):
      agent_executor = initialize_agent(
            self.tools(tools, agents),
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
      )
      return agent_executor.run(query)

  def tool(self):
    def decorator(func):
        self._register_tool(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        return wrapper
    return decorator

  def agent(self, llm, tools=[], agents=[], verbose=False):
    def decorator(func):
        self._register_agent(func.__name__, initialize_agent(
            self.tools(tools, agents),
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        ))

        def wrapper(*args, **kw):
            prompt = func(*args, **kw)
            return self._agents[func.__name__].run(prompt)

        wrapper.__doc__ = func.__doc__
        self._register_tool(wrapper, func.__name__, args_schema=create_schema_from_function(f"{func.__name__}Schema", func))

        return wrapper
    return decorator

