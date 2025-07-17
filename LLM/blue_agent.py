from typing import Dict, Any, List, Optional, TypedDict, Annotated
import logging, os, yaml, json
from dataclasses import dataclass, field
from prettytable import PrettyTable

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results

from LLM.backend import LLMBackend, create_backend
from LLM.configs.prompts import PROMPT_PATH
from LLM.configs.utils import ConfigLoader

# Base LLMAgent, backend_config, create_backend, _build_workflow_graph
logger = logging.getLogger(__name__)
base_path = os.path.dirname(__file__)
base_prompt_path = os.path.join(base_path, "configs", "prompts", PROMPT_PATH)

CAGE2_HOSTS = [
    "User0", "User1", "User2", "User3", "User4", 
    "Enterprise0", "Enterprise1", "Enterprise2", 
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "Defender",
]
CAGE2_ACTIONS = [
    "Monitor",
    "Analyse {host}",
    "Remove {host}",
    "Restore {host}",
    "DecoyApache {host}",
    "DecoyFemitter {host}",
    "DecoyHarakaSMPT {host}",
    "DecoySmss {host}",
    "DecoySSHD {host}",
    "DecoySvchost {host}",
    "DecoyTomcat {host}"
]

def _build_action_mapping():
    mapping = {"Monitor": 0}
    idx = 1
    for action in CAGE2_ACTIONS[1:]:
        for host in CAGE2_HOSTS:
            key = action.format(host=host)
            mapping[key] = idx
            idx += 1
    return mapping

@dataclass
class BlueAgentState:
    messages: Annotated[List, add_messages] = field(default_factory=list)
    current_observation: str = ""
    history: List[str] = field(default_factory=list)
    raw_llm_output: str = ""
    selected_action: Any = None
    episode_step: int = 0
    action_mapping: Dict[str, int] = field(default_factory=dict)

class LLMPolicy:
    def __init__(
        self,
        observation_space, action_space, llm_config
    ):
        self.backend = create_backend(llm_config['llm'], llm_config['hyperparams'])
        
        # CAGE Challenge 2 Specific Things
        self.action_mapping = _build_action_mapping()
        
        # LangGraph workflow
        self.graph = self._build_graph()
        self.state = BlueAgentState(action_mapping=self.action_mapping)

    def get_action(self, observation, action_space=None, hidden=None):
        obs_text = self._vector_to_table(observation)
        self.state.current_observation = obs_text
        self.state.episode_step += 1
        
        output_state = self.graph.invoke(self.state)
        
        if hasattr(output_state, 'selected_action'):
            return output_state.selected_action
        elif isinstance(output_state, dict) and 'selected_action' in output_state:
            return output_state['selected_action']
        else:
            logger.error(f"Unexpected output state format: {output_state}")
            return 0  # Default to Monitor action
    
    def end_episode(self):
        self.state = BlueAgentState(action_mapping=self.action_mapping)
    
    def _build_graph(self):
        logger.info("Build LangGraph Agent")
        # TODO: Add an observation formatter
        graph = StateGraph(BlueAgentState)
        
        # Add nodes
        graph.add_node("format_prompt", self._format_prompt_node)
        graph.add_node("call_llm", self._call_llm_node)
        graph.add_node("parse_action", self._parse_action_node)
        graph.add_node("update_state", self._update_state_node)
        
        # Set entry point
        graph.set_entry_point("format_prompt")
        
        # Add edges
        graph.add_edge("format_prompt", "call_llm")
        graph.add_edge("call_llm", "parse_action")
        graph.add_edge("parse_action", "update_state")
        graph.add_edge("update_state", END)
        
        return graph.compile()
    
    def _format_prompt_node(self, state: BlueAgentState) -> BlueAgentState:
        try:
            prompts = ConfigLoader.load_prompts(base_prompt_path)
            prompt_template = prompts[0]["content"] if prompts else ""
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            prompt_template = ""
            
        # # Format the prompt
        prompt = f"{prompt_template}\n\n# OBSERVATION\n{state.current_observation}\n"
        if state.history:
            prompt += f"\n# HISTORY\n" + "\n".join(state.history)
        
        # Update the current observation with the formatted prompt
        prompt = f"{prompt_template}\n\n# OBSERVATION\n{state.current_observation}\n"
        if state.history:
            prompt += f"\n# HISTORY\n" + "\n".join(state.history)
        state.current_observation = prompt
        return state
    
    def _call_llm_node(self, state: BlueAgentState) -> BlueAgentState:
        # Get the prompt from the format_prompt node
        prompt = state.current_observation
        try:
            # Use the backend to generate a response
            response = self.backend.generate(prompt)
            logger.info("LLM response received")
        except Exception as e:
            logger.error(f"LLM backend error: {e}")
            response = '{"action": "Monitor", "reason": "No valid JSON found in response"}'

        state.raw_llm_output = response
        return state
            
    def _parse_action_node(self, state: BlueAgentState) -> BlueAgentState:
        llm_output = state.raw_llm_output if state.raw_llm_output else ""
        action = None
        # TODO: Parse the LLM output and return the action
        # action = self.action_mapping.get(action_str, 0)  # Default to 0 if not found
        # logger.info(f"Parsed action: {action}")
        state.selected_action = action
        return state
    
    def _vector_to_table(self, observation):
        """
        observation: the numpy array returned by the environment
        """
        HOST_INFO = [
            'Defender',
            'Enterprise0',
            'Enterprise1',
            'Enterprise2',
            'Op_Host0',
            'Op_Host1',
            'Op_Host2',
            'Op_Server0',
            'User0',
            'User1',
            'User2',
            'User3',
            'User4',
        ]
        table = PrettyTable(['Hostname', 'Activity', 'Compromised'])
        idx = 0
        for host in HOST_INFO:
            # Activity: 2 bits
            activity_bits = observation[idx:idx+2]
            if (activity_bits == [0,0]).all():
                activity = 'None'
            elif (activity_bits == [1,0]).all():
                activity = 'Scan'
            elif (activity_bits == [1,1]).all():
                activity = 'Exploit'
            else:
                activity = 'Unknown'
            idx += 2

            # Compromised: 2 bits
            comp_bits = observation[idx:idx+2]
            if (comp_bits == [0,0]).all():
                compromised = 'No'
            elif (comp_bits == [1,0]).all():
                compromised = 'Unknown'
            elif (comp_bits == [0,1]).all():
                compromised = 'User'
            elif (comp_bits == [1,1]).all():
                compromised = 'Privileged'
            else:
                compromised = 'Unknown'
            idx += 2

            table.add_row([host, activity, compromised])
        return table
    
    def _observation_to_text(self, observation):
        # Convert numpy array to a more readable format
        if hasattr(observation, 'tolist'):
            return str(observation.tolist())
        return str(observation)
    
    def _update_state_node(self, state: BlueAgentState) -> BlueAgentState:
        # Update history and state for next step
        if state.raw_llm_output:
            state.history.append(state.raw_llm_output)
        return state

class LLMAgent:
    def __init__(self, name, policy, obs_space, llm_config):
        self.policy = policy(obs_space, None, llm_config)
        self.obs_space = obs_space
        self.end_episode()
        
    def get_action(self, observation, action_space=None):
        action = self.policy.get_action(observation)
        self.step += 1
        return action
    
    def end_episode(self):
        self.step = 0
        self.last_action = None