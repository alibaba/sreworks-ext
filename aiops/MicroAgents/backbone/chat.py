
import openai
import requests
import dashscope
from dashscope import Generation
from dashscope.aigc.generation import AioGeneration
from logger import global_logger
from http import HTTPStatus
import json
import os
import asyncio
import openai

LOCALNAME = {
  'qwen-7B':'./Qwen1.5-7B-Chat',
  'qwen-14B':'./Qwen1.5-14B-Chat',
  'llama3': './Meta-Llama-3-8B-Instruct',
  'mistral-7B': './Mistral-7B-Instruct-v0.2'
}
class OpenaiChat:
  def __init__(self, llm_name, temperature=1.,top_p=1e-7,max_tokens=2000,stop=None, tools=None, **kwargs):
    self.llm_name = llm_name
    self.temperature = temperature
    self.top_p = top_p
    self.max_tokens = max_tokens
    self.stop = stop
    self.tools = tools
    self.kwargs = kwargs
    
    
  def reply(self, messages, tools=None, **kwargs):
    openai_model = LOCALNAME[self.llm_name]
    openai.api_key= "EMPTY"
    openai.api_base = "http://127.0.0.1:8000/v1"
   
    client = openai.OpenAI(
      base_url="http://localhost:8000/v1",
      api_key="EMPTY",  
      )
    if 'mistral' in self.llm_name and 'system' in messages[0]['role']:
      messages[1]['content'] = messages[0]['content'] +'\n'+ messages[1]['content']
      messages = messages[1:]
      
      
    response = client.chat.completions.create(
      model=openai_model,
      messages=messages,
      temperature=self.temperature,top_p=self.top_p, max_tokens=self.max_tokens, stop=self.stop
    )
    
    
    global_logger.global_logger.input_token_nums += response.usage.prompt_tokens
    global_logger.global_logger.output_token_nums += response.usage.completion_tokens
    
    return response, True
  
  async def areply(self, messages, tools=None, **kwargs):
    openai_model = LOCALNAME[self.llm_name]
    openai.api_key= "EMPTY"
    openai.api_base = "http://127.0.0.1:8000/v1"
   
    client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  
    )

    response = await client.chat.completions.create(
      model=openai_model,
      messages=messages,
      temperature=self.temperature,top_p=self.top_p, max_tokens=self.max_tokens, stop=self.stop
    )
    
    global_logger.global_logger.input_token_nums += response.usage.prompt_tokens
    global_logger.global_logger.output_token_nums += response.usage.completion_tokens
    
    return response, True
  

class Qwen:
  def __init__(self, llm_name='qwen-max', temperature=1.,top_p=1e-7,max_tokens=2000,stop=None, tools=None, **kwargs):
    self.qwen_model = llm_name
    self.temperature = temperature
    self.top_p = top_p
    self.max_tokens = max_tokens
    self.stop = stop
    self.tools = tools
    self.kwargs = kwargs
    
  def reply(self, messages, tools=None, **kwargs):
    # dashscope.api_key= os.environ['TONGYI_KEY']
    with open('./common/configs/key.json', 'r') as f:
      keys = json.load(f)
      dashscope.api_key= keys['key']

    if tools is None and not self.tools is None:
      tools  = self.tools
    model = Generation.Models.qwen_max if self.qwen_model == 'qwen-max' else Generation.Models.qwen_turbo
    print(f'Using model:{self.qwen_model}', model)
    response=Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        top_p=self.top_p,
        temperature=self.temperature,
        stop=self.stop, 
        max_tokens=self.max_tokens,
        tools = tools
    )
    if response.status_code==HTTPStatus.OK:
        text= response.output
        global_logger.global_logger.input_token_nums += response.usage.input_tokens
        global_logger.global_logger.output_token_nums += response.usage.output_tokens
    else:
        text= 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
    # print(text, '\n')
    

    return text, response.status_code==HTTPStatus.OK
  
  async def areply(self, messages, tools=None, **kwargs):
    with open('./common/configs/key.json', 'r') as f:
        keys = json.load(f)
        dashscope.api_key= keys['key']
    if tools is None and not self.tools is None:
      tools  = self.tools
      
    model = Generation.Models.qwen_max if self.qwen_model == 'qwen_max' else Generation.Models.qwen_turbo

    response = await AioGeneration.call(
          model=model,
          messages=messages,
          result_format='message',
          top_p=self.top_p,
          temperature=self.temperature,
          stop=self.stop,
          max_tokens=self.max_tokens,
          tools = tools
      )
    if response.status_code==HTTPStatus.OK:
        text= response.output
        global_logger.global_logger.input_token_nums += response.usage.input_tokens
        global_logger.global_logger.output_token_nums += response.usage.output_tokens
    else:
        text= 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
    # print(text, '\n')

    return text, response.status_code==HTTPStatus.OK


