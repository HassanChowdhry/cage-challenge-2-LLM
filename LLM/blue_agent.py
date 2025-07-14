from typing import Dict, Any, List, Optional, TypedDict, Annotated
import logging
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from .base import BaseLLMAgent
from .backends import LLMBackend, create_backend
from .prompts import get_prompt_template

# Base LLMAgent, backend_config, create_backend, _build_workflow_graph
logger = logging.getLogger(__name__)

@dataclass
class BlueAgentState:
    messages: Annotated[List, add_messages] = field(default_factory=list)
    current_observation: str = ""
    history: List[str] = field(default_factory=list)
    summary: str = ""
    raw_llm_output: str = ""
    selected_action: Any = None
    episode_step: int = 0
    action_mapping: Dict[str, int] = field(default_factory=dict)

class LLMBlueAgent(BaseLLMAgent):    
    def __init__(
        self,
        backend_type: str = "gemini",
        hyperparams: Dict[str, Any] = None,
        prompt_name: str = "zero_shot",
        max_history_length: int = 100,
    ):
        self.backend = create_backend(backend_type)(hyperparams)
        
        # config
        self.max_history_length = max_history_length
        
        self.prompt_name = prompt_name
        self.prompt_template = get_prompt_template(prompt_name)
        
        # CAGE Challenge 2 Specific Things
        self.action_mapping = self._build_action_mapping()
        
        # LangGraph workflow
        self.graph = self._build_graph()
        self.state = BlueAgentState(action_mapping=self.action_mapping)

    def _build_action_mapping(self) -> Dict[str, int]:
        actions = []
        hosts = ["User0", "User1", "User2", "Enterprise0", "Enterprise1", "Enterprise2", "Operational0"]
        
        # Spreads this into Action *host0* format
        for host in hosts:
            actions += [
                f"Analyze {host}",
                f"Remove {host}", 
                f"Restore {host}",
                f"Decoy {host}"
            ]
        # Puts it in Analyse *host*: 0,...
        action_mapping = {action: idx for idx, action in enumerate(actions)}
        return action_mapping
    
    def _build_graph(self):
        logger.info("Build LangGraph Agent")

        graph = StateGraph(BlueAgentState)
        
        # Create Nodes in LangGraph
        graph.add_node("format_prompt", self._format_prompt_node)
        graph.add_node("call_llm", self._call_llm_node)
        graph.add_node("parse_action", self._parse_action_node)
        graph.add_node("update_state", self._update_state_node)
        
        # Add edges
        graph.set_entry_point("format_prompt")
        graph.add_edge("format_prompt", "call_llm")
        graph.add_edge("call_llm", "parse_action")
        graph.add_edge("parse_action", "update_state")
        graph.add_edge("update_state", END)
        
        return workflow.compile()
    
    def _format_prompt_node(self, state: BlueAgentState) -> Dict[str, Any]:
        pass
    
    def _call_llm_node(self, state: BlueAgentState) -> Dict[str, Any]:
        pass
            
    
    def _parse_action_node(self, state: BlueAgentState) -> Dict[str, Any]:
        pass
    
    def _update_state_node(self, state: BlueAgentState, parsed_action_text: str) -> Dict[str, Any]:
        step_ = f"Step {state.episode_step}: {parsed_action_text}"
        state.history.append(step_)
        
        if len(state.history) > self.max_history_length:
            state.history = state.history[-self.max_history_length:]
        
        recent_actions = state.history[-3:] if len(state.history) >= 3 else state.history
        state.summary = "; ".join(recent_actions)
        
        state.episode_step += 1
        
        return {}
    
    def _observation_to_text(self, observation) -> str:
        if isinstance(observation, str):
            return observation
        
        if isinstance(observation, (list, tuple)):
            return self._vector_observation_to_text(observation)
        
        if isinstance(observation, dict):
            return self._dict_observation_to_text(observation)
        
        return str(observation)
    
    def _vector_observation_to_text(self, obs_vector) -> str:
        # This should be customized based on the actual CAGE observation format
        # For now I'll put a generic description
        return f"Network observation vector with {len(obs_vector)} features"
    
    def _dict_observation_to_text(self, obs_dict) -> str:
        alerts = []
        for key, value in obs_dict.items():
            if value and key.lower() in ['alert', 'compromise', 'malware', 'suspicious', 'detected']:
                alerts.append(f"{key}: {value}")
        
        if alerts:
            return "; ".join(alerts)
        else:
            return "No alerts detected"
    
    def get_action(self, observation, action_space=None, hidden=None):
        obs_text = self._observation_to_text(observation)
        self.state.current_observation = obs_text

        logger.info(f"Invoke LangGraph")
        result = self.graph.invoke(self.state)
        action = result.selected_action

        return action
    
    def train(self, results: Results):
        self._log_transition(results.observation, results.action, results.reward) # store transition
    
    def end_episode(self):
        self.state = BlueAgentState(action_mapping=self.action_mapping)
        super().end_episode()
    
    def set_initial_values(self, action_space, observation):
        pass
    

# def _call_llm_node(self, state: BlueAgentState) -> Dict[str, Any]:
#     try:
#         # Get the last message (should be the system message)
#         if state.messages:
#             last_message = state.messages[-1]
#             prompt = last_message.content
#         else:
#             prompt = self.prompt_template.format(
#                 summary=state.summary or "No previous actions",
#                 observation=state.current_observation
#             )
        
#         response = self.backend.generate(prompt)
        
#         ai_message = AIMessage(content=response)
        
#         return {"messages": [ai_message], "raw_llm_output": response}
        
#     except Exception as e:
#         logger.error(f"LLM generation error: {e}")
#         fallback_response = "Analyze User0"
#         ai_message = AIMessage(content=fallback_response)
#         return {"messages": [ai_message], "raw_llm_output": fallback_response}

# def _format_prompt_node(self, state: BlueAgentState) -> Dict[str, Any]:
#     [system_content = self.prompt_template.format(
#         summary=state.summary or "No previous actions",
#         observation=state.current_observation
#     )
    
#     system_message = SystemMessage(content=system_content)
    
#     return {"messages": [system_message]}]

# def _call_llm_node(self, state: BlueAgentState) -> Dict[str, Any]:

#     # Get the last message (should be the system message)
#     if state.messages:
#         last_message = state.messages[-1]
#         prompt = last_message.content
#     else:
#         prompt = self.prompt_template.format(
#             summary=state.summary or "No previous actions",
#             observation=state.current_observation
#         )
    
#     response = self.backend.generate(prompt)
    
#     ai_message = AIMessage(content=response)
    
#     return {"messages": [ai_message], "raw_llm_output": response}

# def _parse_action_node(self, state: BlueAgentState) -> Dict[str, Any]:
#     if state.messages:
#         ai_message = state.messages[-1]
#         action_text = ai_message.content.strip()
#     else:
#         action_text = state.raw_llm_output.strip()
    
#     for action_name in self.action_mapping.keys():
#         if action_name.lower() in action_text.lower():
#             action_idx = self.action_mapping[action_name]
#             return {"selected_action": action_idx, "parsed_action_text": action_name}
    
#     # If no exact match, try to parse common patterns
#     words = action_text.split()
#     if len(words) >= 2:
#         action_type = words[0].capitalize()
#         host_name = words[1]
        
#         # Try to find a matching action
#         for action_name in self.action_mapping.keys():
#             if action_name.startswith(action_type) and host_name in action_name:
#                 action_idx = self.action_mapping[action_name]
#                 return {"selected_action": action_idx, "parsed_action_text": action_name}
    
#     # Fallback
#     logger.warning(f"Could not parse action from: {action_text}")
#     return {"selected_action": 0, "parsed_action_text": "Analyze User0"}