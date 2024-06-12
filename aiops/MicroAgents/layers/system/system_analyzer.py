import os
from typing import Callable, Dict, Literal, Optional, Union
from functools import partial
from agents import AgentBase, ExpertAgent, ModuleAgent      
import networkx as nx
from ..utils import get_k_order_neighbors_with_direction
import copy
from logger.logger import Logger
import asyncio

class SystemAnalyzer:
    def __init__(self, modules, model, system_instructions, tools=None, human_input_mode=False, intervention_mode: Literal["COVER", "APPEND"] = "COVER"):
        """
        Args:
            modules (list[dict]): A list of modules. Each module is a dict with keys: "module_name", "module_function", "module dependency"
            model_config_name (str): The configuration file name of the model.
            system_instructions (str): The system prompt for the module agent.
            tools (dict): The tools for the module agent.
            human_input_mode (bool): Whether to enable human input mode. If True, the system will ask the human expert for the diagnosis result.
            intervention_mode (Literal["COVER", "APPEND"]): The intervention mode. COVER means the system will cover the diagnosis result with the human expert's analysis; APPEND means the system will append the human expert's analysis to the diagnosis result.
        """
        self.modules = modules
        self.agents = []
        self.agents_name = {}
        self.human_input_mode = human_input_mode
        self.intervention_mode = intervention_mode

        self.huamn_expert = ExpertAgent(name="human expert", human_input_mode=human_input_mode)
        for module in self.modules.values():
            module_analyzer = ModuleAgent(name=f"{module['module_name']} agent", tools=tools,module=module, model=model, system_instructions=system_instructions)
            self.agents.append(module_analyzer)
            self.agents_name[module["module_name"]] = self.agents[-1]
        
    def analyze(self, graph, modules, organization_mode:Literal["forward", "backward", "forward_and_backward", "convolution", "isolation", "full"]="backward", k:int=1, layers=1):
        '''
        根据系统的依赖关系，分析整个系统的异常机器根因，可以从模块依赖图前向分析、后向分析、先前向再后向分析或者类似图神经网络进行图扩散分析（同时考虑前继节点和后继节点）
        Args:
            graph (dict): A networkx graph of the system indicating the dependencies between modules. The graph must be acyclic.
            modules (dict): A dict of modules. Each module is a dict with keys: "module_name", "module_function", "symptom", "module_dependency"
        '''
        if organization_mode in ["forward", "backward", "isolation", "full"]:
            decisions = self.directional_analyze(graph, modules, direction=organization_mode, decisions=None, k=k)

        elif organization_mode =="forward_and_backward":
            decisions = self.forward_and_backward_analyze(graph, modules, k=k, layers=layers)
        elif organization_mode =="convolution":
            decisions = self.convolution_analyze(graph, modules, k=k, layers=layers)
        else:
            raise NotImplementedError(f"organization mode {organization_mode} is not implemented !!!")
            decisions = None

        if self.human_input_mode:
            for node in decisions.keys():
                if decisions[node]:
                    human_response = self.huamn_expert.reply(decisions[node])
                    if self.intervention_mode.lower() == 'cover' and human_response:
                        decisions[node] = human_response
                    elif self.intervention_mode.lower() == 'append' and human_response:
                        decisions[node] += f"\nHuman expert's analysis: {human_response}"
        return decisions


    def filter_nodes(self, graph, node_list, node_anomaly_map):
        '''
        过滤掉邻居节点和自身均正常的节点
        '''
        node_list_filtered = copy.deepcopy(node_list)
       
        for node in node_list:
            if not node_anomaly_map[node]["symptom"]:
                node_list_filtered.remove(node)
        return node_list_filtered
    
    async def a_directional_analyze(self, graph, modules, direction:Literal["forward", "backward", "bidirectional", "isolation", "full"]="forward", decisions=None, k:int=1):
        """
        遍历一次进行分析，可以在整个模块依赖图上进行一次前向、后向或者双向传播
        Args:
            graph (dict): A networkx graph of the system indicating the dependencies between modules. The graph must be acyclic.
            modules (dict): A dict of modules. Each module is a dict with keys: "module_name", "module_function", "symptom"。
            k (int): consider the k-order neighbors.
        """
        if decisions is None:
            decisions = {}
            for node in modules:
                if modules[node]["symptom"]:
                    decisions.update({node: modules[node]["symptom"]})

        if direction == "bidirectional": #如果是双向传播，即图卷积分析，在整层分析完后再进行decisions的更新
            cur_decisions = copy.deepcopy(decisions)
        else:
            cur_decisions = decisions

        if direction == "backward":
            sorted_nodes = list(nx.topological_sort(graph))[::-1]
        else:
            sorted_nodes = list(nx.topological_sort(graph))
        sorted_nodes = self.filter_nodes(graph, sorted_nodes, modules)

        loggers = []
        for node in sorted_nodes:
            logger = Logger()
            logger['cur_node'] = node
            logger['symptom'] = modules[node]["symptom"]
            logger['relations'] = modules[node]["module_dependency"]
            logger['direction'] = direction
            logger['neighbors'] = []

            agent = self.agents_name[node]

            if direction == "full":
                k = 1
                neighbor_nodes = {i: list(graph.nodes()) for i in range(1, k+1)}
            else:
                neighbor_nodes = get_k_order_neighbors_with_direction(graph, node, k, direction)

            message_from_other_agent = ""
            count = 1
            if direction == "isolation":
                logger['neighbors'] = []
            else:
                for i in range(1, k+1):
                    i_order_neighbors = neighbor_nodes[i]
                    if i_order_neighbors:
                        for neighbor in list(i_order_neighbors):
                            if neighbor in cur_decisions and cur_decisions[neighbor]:
                                # message_from_other_agent += f"""{count}. Diagnosis of the module {modules[neighbor]["module_name"]}. \n Module function: {modules[neighbor]["module_function"]}. \n Diagnosis result: {cur_decisions[neighbor]} \n"""
                                count += 1

                                logger['neighbors'].append({"module_name": modules[neighbor]["module_name"],"module_function": modules[neighbor]["module_function"], "diagnosis_result": cur_decisions[neighbor]})
            
            if modules[node]["symptom"] or logger['neighbors']:
                
                loggers.append(logger)
            else:
                response = ""
                decisions.update({node: response})
        start = 0
        end = 0
        while start < len(loggers):
            start = end
            end = start + 4
            if end > len(loggers):
                end = len(loggers)
            responses = await asyncio.gather(*[self.agents_name[logger['cur_node']].areply(logger.to_message_text(direction)) for logger in loggers[start:end]])
            
            for response, logger in zip(responses, loggers[start:end]):
                print('-'*25, f'{logger["cur_node"]}', '-'*25)
                logger.print()
                print(f'Diagnosis result of the module {logger["cur_node"]} in the current layer with the {direction} direction:\n', response)
                decisions.update({logger['cur_node']: response})
            
        return decisions



    def directional_analyze(self, graph, modules, direction:Literal["forward", "backward", "bidirectional", "isolation", "full"]="forward", decisions=None, k:int=1):
        """
        遍历一次进行分析，可以在整个模块依赖图上进行一次前向、后向或者双向传播
        Args:
            graph (dict): A networkx graph of the system indicating the dependencies between modules. The graph must be acyclic.
            modules (dict): A dict of modules. Each module is a dict with keys: "module_name", "module_function", "symptom"。
            k (int): consider the k-order neighbors.
        """
        if decisions is None:
            decisions = {}
            for node in modules:
                if modules[node]["symptom"]:
                    decisions.update({node: modules[node]["symptom"]})

        if direction == "bidirectional": #如果是双向传播，即图卷积分析，在整层分析完后再进行decisions的更新
            cur_decisions = copy.deepcopy(decisions)
        else:
            cur_decisions = decisions

        if direction == "backward":
            sorted_nodes = list(nx.topological_sort(graph))[::-1]
        else:
            sorted_nodes = list(nx.topological_sort(graph))
        sorted_nodes = self.filter_nodes(graph, sorted_nodes, modules)

        logger = Logger()
        for node in sorted_nodes:
            logger['cur_node'] = node
            logger['symptom'] = modules[node]["symptom"]
            logger['relations'] = modules[node]["module_dependency"]
            logger['direction'] = direction
            logger['neighbors'] = []

            agent = self.agents_name[node]

            if direction == "full":
                k = 1
                neighbor_nodes = {i: list(graph.nodes()) for i in range(1, k+1)}
            else:
                neighbor_nodes = get_k_order_neighbors_with_direction(graph, node, k, direction)

            message_from_other_agent = ""
            count = 1
            if direction == "isolation":
                logger['neighbors'] = []
            else:
                for i in range(1, k+1):
                    i_order_neighbors = neighbor_nodes[i]
                    if i_order_neighbors:
                        for neighbor in list(i_order_neighbors):
                            if neighbor in cur_decisions and cur_decisions[neighbor]:
                                # message_from_other_agent += f"""{count}. Diagnosis of the module {modules[neighbor]["module_name"]}. \n Module function: {modules[neighbor]["module_function"]}. \n Diagnosis result: {cur_decisions[neighbor]} \n"""
                                count += 1

                                logger['neighbors'].append({"module_name": modules[neighbor]["module_name"],"module_function": modules[neighbor]["module_function"], "diagnosis_result": cur_decisions[neighbor]})

            if modules[node]["symptom"] or logger['neighbors']:
                print('-'*25, f'{node}', '-'*25)
                logger.print()
                response = agent.reply(logger.to_message_text(direction))
                print(f'Diagnosis result of the module {node} in the current layer with the {direction} direction:\n', response)
            else:
                response = ""
            decisions.update({node: response})
        

        return decisions

    def forward_and_backward_analyze(self, graph, modules, k=1, layers=1):

        """
        遍历两次进行分析，先前向再后向传播
        Args:
            graph (dict): A networkx graph of the system indicating the dependencies between modules. The graph must be acyclic.
            modules (dict): A dict of modules. Each module is a dict with keys: "module_name", "module_function", "symptom", "module_dependency"
        """
        decisions = {}
        for node in modules:
            if modules[node]["symptom"]:
                decisions.update({node: modules[node]["symptom"]})
        for _ in range(layers):
            decisions = self.directional_analyze(graph, modules, direction="forward", decisions=decisions, k=k)
            decisions = self.directional_analyze(graph, modules, direction="backward", decisions=decisions, k=k)
        return decisions
    
    def convolution_analyze(self, graph, modules, k=1, layers=1):
        """
        遍历多次进行分析，类似图神经网络进行图扩散分析（同时考虑前继节点和后继节点）
        Args:
            graph (dict): A networkx graph of the system indicating the dependencies between modules. The graph must be acyclic.
            modules (dict): A dict of modules. Each module is a dict with keys: "module_name", "module_function", "symptom", "module_dependency"
        """
        decisions = {}
        for node in modules:
            if modules[node]["symptom"]:
                decisions.update({node: modules[node]["symptom"]})
        for __ in range(layers):
            print('='*25, f'layer {__}', '='*25)
            # decisions = asyncio.run(self.a_directional_analyze(graph, modules, direction="bidirectional", decisions=decisions, k=k))
            decisions = self.directional_analyze(graph, modules, direction="bidirectional", decisions=decisions, k=k)
        return decisions

    
    




    
    
        






            

        
        
        