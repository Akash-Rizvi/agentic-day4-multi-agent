from langchain_core.messages import SystemMessage, HumanMessage
from state import MultiAgentState
import yaml
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime

load_dotenv()

with open("prompts/supervisor_v1.yaml", "r", encoding="utf-8") as f:
    prompt_data = yaml.safe_load(f)

supervisor_system_prompt_from_yaml = prompt_data["system"]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )


def supervisor_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content=supervisor_system_prompt_from_yaml),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()

    if route not in VALID_ROUTES:
        route = "general"

    handoff = AgentHandoff(
        from_agent="supervisor",
        to_agent=route,
        task=state["user_request"],
        context={"route": route},
        priority="normal",
        timestamp=datetime.utcnow().isoformat(),
    )

    return {
        "route": route,
        "handoff_context": handoff.to_prompt_context(),
    }



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