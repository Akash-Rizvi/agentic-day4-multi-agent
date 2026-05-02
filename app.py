from agents import (
    AgentHandoff,
    orders_agent_node,
    billing_agent_node,
    technical_agent_node,
    subscription_agent_node,
    general_agent_node,
    synthesize_response_node,
)
from typing import TypedDict
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal
from langgraph.graph import StateGraph, END
import re
from typing import Final
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from audit import SessionAuditLog, persist_audit_log
from datetime import datetime
from agents import AgentHandoff
load_dotenv()

from typing import TypedDict

class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user
    handoff_context: str

with open("prompts/supervisor_v1.yaml", "r", encoding="utf-8") as f:
    prompt_data = yaml.safe_load(f)

supervisor_system_prompt_from_yaml = prompt_data["system"]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]

def supervisor_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content=supervisor_system_prompt_from_yaml),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}

def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")

def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")

    workflow.add_conditional_edges(
        "supervisor_node",
        route_to_specialist,
    )

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()

def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input

def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")

    with open("prompts/supervisor_v1.yaml", "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)
    _ = prompt_data["system"]

    graph = build_graph()

    for request in [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]:
        safe_text = guard_request(request)

        if safe_text != request:
            print("Request:", request)
            print("Route:", "blocked", "Agent used:", "guard_request")
            print("Final:", safe_text)
            print("---")
            audit.log(agent="guard_request", action="blocked_request", tokens_in=0, tokens_out=0)
            continue

        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
            "handoff_context": "",
        }

        result = graph.invoke(state)

        print("Request:", request)
        print("Route:", result.get("route"), "Agent used:", result.get("agent_used"))
        print("Final:", result.get("final_response"))
        print("---")

        audit.log(
            agent=result.get("agent_used", "unknown_agent"),
            action=result.get("route", "unknown_route"),
            tokens_in=100,
            tokens_out=50,
        )

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)


if __name__ == "__main__":
    main()