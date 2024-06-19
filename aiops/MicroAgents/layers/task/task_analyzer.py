
from agents import AgentBase, GroupLeader, TaskAgent

            
class TaskAnalyzer:
    

    def __init__(self, use_memory, memory_config, model, system_instructions_task_agent, user_message_task_agent, system_instructions_group_leader, user_message_group_leader, tools=None, num_agents=3, human_input_mode=False):
        
        self.num_agents = num_agents
        self.agents = []
        for i in range(num_agents):
            self.agents.append(TaskAgent(f"SRE {i}", use_memory, model, memory_config, system_instructions_task_agent, user_message_task_agent, tools=tools))
        self.human_input_mode = human_input_mode
        self.group_leader = GroupLeader(f"group_leader", use_memory, model, memory_config, system_instructions_group_leader, user_message_group_leader, tools=tools)
    
    def analyze(self, system_info, critic_info=[], chat=False, chat_turn=1):
        
        result_list = []
        for _ in range(chat_turn):
            for i, agent in enumerate(self.agents):
                memory_text = ""
                if chat:
                    for i, m in enumerate(result_list):
                        memory_text += f"SRE_{i%self.num_agents}: {m}\n"
                response = agent.reply(system_info, critic_info=critic_info, externel_message=memory_text)
                result_list.append(response)
                
                print("-"*50)
                print(f"Analysis results from {agent.name}: \n{response}\n")
        return self.merge_result(result_list)
    
    def merge_result(self, result_list):
        if len(result_list) == 1:
            return result_list[0]
            
        analysis_results = ""
        for i, res in enumerate(result_list):
            analysis_results += f"Analysis results from SRE {i%self.num_agents}: \n{res}\n"
        conclusion = self.group_leader.reply(analysis_results)

        return conclusion

        
        

        
        


