from layers.data_layer import Log, Metric, Trace, DataAnalyzer
from layers.system.system_analyzer import SystemAnalyzer
from layers.task.task_analyzer import TaskAnalyzer
import polars as pl
from utils.process import get_network_metrics
import random
from utils.rag import RAG

class AgentNN:
    def __init__(self, system_intro, architecture, use_memory, memory_config, model, data_type=["log", "trace", "metric"], prompts={},
                 template_file="", config_file="", log_result_file="", num_agents=3, human_input_mode=False, intervention_mode="COVER", rag=False):

        modules_info = {}
        for module, description in architecture.nodes(data=True):
            modules_info[module] = {
                "module_name": module,
                "module_function": description,
                "module_dependency": {
                    "callers": list(architecture.successors(module)),
                    "callees": list(architecture.predecessors(module)),
                }
            }


        self.prompts = prompts
        system_instructions_system_analyzer = prompts["system_analyzer"]
        system_instructions_task_agent, user_message_task_agent = prompts["task_analyzer"]['task_agent']
        system_instructions_group_leader, user_message_group_leader = prompts["task_analyzer"]['group_leader']



        self.system_intro = system_intro
        self.task_layer = TaskAnalyzer(use_memory, memory_config, model=model, num_agents=num_agents, human_input_mode=human_input_mode, system_instructions_task_agent=system_instructions_task_agent, user_message_task_agent=user_message_task_agent, system_instructions_group_leader=system_instructions_group_leader, user_message_group_leader=user_message_group_leader)
        self.system_layer = SystemAnalyzer(modules_info, model=model, system_instructions=system_instructions_system_analyzer,human_input_mode=human_input_mode, intervention_mode=intervention_mode)
        self.data_layer = DataAnalyzer(data_type=data_type,template_file=template_file, config_file=config_file, log_result_file=log_result_file, model=model)
        self.architecture = architecture
        self.modules_info = modules_info

        if rag:
            self.rag = RAG()
        else:
            self.rag = None
    
    def analyze(self, history_df_logs, history_df_traces, history_df_metrics, recent_df_logs, recent_df_traces, recent_df_metrics, metric_names=[], 
                organization_mode="backward",
                time_col='m_timestamp', log_col='m_message', window_size={'minutes':1}, k=3, summary=False, only_ad=False, order=1):
        
        print("#"*25, "DATA ANALYSIS", "#"*25)
        print(history_df_logs.shape)
        history_df_logs = history_df_logs.filter(pl.col(log_col).str.lengths()<=1000)
        similar_incidents = []
        for module, description in self.architecture.nodes(data=True):

            module_history_df_log = history_df_logs.filter(pl.col('PodName') == module)
            module_history_df_trace = history_df_traces.filter(pl.col('PodName') == module)
            module_history_df_metric = history_df_metrics.filter(pl.col('PodName') == module)
            

            module_recent_df_log = recent_df_logs.filter(pl.col('PodName') == module).filter(pl.col(log_col).str.lengths()<=1000)
            # module_recent_df_trace = recent_df_traces.filter(pl.col('PodName') == module)
            module_recent_df_trace = get_network_metrics(recent_df_traces, pod_name=module)
            module_recent_df_metric = recent_df_metrics.filter(pl.col('PodName') == module)

            logs = {
                "history_df": history_df_logs,
                "recent_df": module_recent_df_log,
                "time_col": time_col,
                "log_col": log_col,
                "window_size": window_size,
                "k": k,
                "summary": summary
            }

            traces = {
                "history_df": history_df_traces,
                "recent_df": module_recent_df_trace,
                "metric_columns": ["NetworkP90(ms)"],
                "k": k,
            }

            metrics = {
                "history_df": history_df_metrics,
                "recent_df": module_recent_df_metric,
                "metric_columns": metric_names,
                "k": k,
            }

            # if module_recent_df_log.shape[0] > 0:
            module_data_analysis_results, symptoms = self.data_layer.analyze(logs, traces, metrics, use_rule=False)
            # else:
            #     module_data_analysis_results, symptoms = "", []
            self.modules_info[module]['symptom'] = module_data_analysis_results
            self.modules_info[module]['symptom_list'] = symptoms
            if module_data_analysis_results:
                print('-'*25, f"MODULE {module} ANALYSIS", '-'*25)
                print(f"{module_data_analysis_results}\n")

            if (not self.rag is None) and module_data_analysis_results:
                similar_incident = self.rag.retrieve(symptoms)[:2]
                if similar_incident:
                    similar_incidents.extend(similar_incident)
        if (not self.rag is None) and len(similar_incidents)>0:
            similar_incidents = self.rag.retrieve_to_text(similar_incidents)
            print('Get similar incidents: \n', similar_incidents, '\n')
        else:
            similar_incidents = ""


        if only_ad:
            return ""


        print("#"*25, "SYSTEM ANALYSIS", "#"*25)
        system_analysis_results = self.system_layer.analyze(self.architecture, self.modules_info, organization_mode=organization_mode, k=1, layers=2)

        module_analysis_text = "The analysis results of the system are as follows: \n"
        count = 1


        if order==1:
            keys = system_analysis_results.keys()
        elif order == -1:
            keys = list(system_analysis_results.keys())[::-1]
        else:
            keys = list(system_analysis_results.keys())
            random.shuffle(keys)

        for key in keys:
            cur_analysis_text = (f"{count}. Module Name: {key}; \n "
                                     f"Module Function: {self.modules_info[key]['module_function']}; \n "  
                                     f"Module Dependency: {self.modules_info[key]['module_dependency']}; \n"
                                     f"Expert analysis result of module {key}: {system_analysis_results[key]}\n")

            if system_analysis_results[key]:
                print("-"*25, f'{count}. {key}', "-"*25)
                module_analysis_text += cur_analysis_text
                count += 1
                print(f"{cur_analysis_text}\n")
                


        
        module_list = ""
        count = 1
        for module in self.modules_info:
            if self.modules_info[module]['symptom']:
                if count > 10:
                    module_list += f"and etc."
                    break
                module_list += f"{module}, "
                count += 1
        
       
        system_info = {"system_introduction":self.system_intro,
                        "module_list": module_list,
                        "module_analysis": module_analysis_text,
                        "similar_incidents": similar_incidents}

        print("#"*25, "TASK ANALYSIS", "#"*25)

        task_analysis_results = self.task_layer.analyze(system_info)

        print(f"{task_analysis_results}")


        return task_analysis_results, module_analysis_text


