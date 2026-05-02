from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:

    from app import MultiAgentState


@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str  # "low" | "normal" | "high"
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )


def orders_agent_node(state: MultiAgentState) -> dict:
    text = f"[orders_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "orders_agent",
        "specialist_result": text,
    }
def billing_agent_node(state: MultiAgentState) -> dict:
    text = (
        f"{state['handoff_context']}\n\n"
        f"[billing_agent] Handling request: {state['user_request']}"
    )
    return {
        "agent_used": "billing_agent",
        "specialist_result": text,
    }
def technical_agent_node(state: MultiAgentState) -> dict:
    text = f"[technical_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "technical_agent",
        "specialist_result": text,
    }
def subscription_agent_node(state: MultiAgentState) -> dict:
    text = f"[subscription_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "subscription_agent",
        "specialist_result": text,
    }
def general_agent_node(state: MultiAgentState) -> dict:
    text = f"[general_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "general_agent",
        "specialist_result": text,
    }
def synthesize_response_node(state: MultiAgentState) -> dict:
    final_text = (
        f"Route selected: {state['route']}\n"
        f"Agent used: {state['agent_used']}\n"
        f"Response: {state['specialist_result']}"
    )
    return {
        "final_response": final_text,
    }
    
