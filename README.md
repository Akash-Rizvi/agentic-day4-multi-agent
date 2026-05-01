# agentic-day4-multi-agent

Day 4 assignment project for **Multi-Agent Collaboration – Supervisor + Specialists**.

This project implements a production-minded **multi-agent customer support system** using **LangGraph** and **LangChain** chat models. It includes:

- a **supervisor agent** for routing
- **specialist agents** for different domains
- **structured handoffs** between agents
- **prompt injection detection** at graph entry
- **session audit logging with cost tracking**

## Project Structure

```text
agentic-day4-multi-agent/
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
├── agents.py
├── audit.py
├── fallbacks.py
├── state.py
└── prompts/
    └── supervisor_v1.yaml
```

## Features Implemented

### 1. Supervisor Agent + Routing
- The supervisor prompt is stored in:
  - `prompts/supervisor_v1.yaml`
- `supervisor_node` reads `user_request` from `MultiAgentState`
- The supervisor uses an LLM to classify requests into one route:
  - `orders`
  - `billing`
  - `technical`
  - `subscription`
  - `general`
- The graph uses `StateGraph` and `add_conditional_edges` to route to the right specialist

### 2. Specialist Agents
The project includes specialist nodes:
- `orders_agent_node`
- `billing_agent_node`
- `technical_agent_node`
- `subscription_agent_node`
- `general_agent_node`

Each specialist:
- reads `user_request`
- sets `agent_used`
- writes `specialist_result`

### 3. Structured Handoffs
- `AgentHandoff` is implemented as a dataclass
- The supervisor creates a typed handoff before passing work to a specialist
- Handoff data is carried as auditable context

### 4. Injection Detection at Graph Entry
- `detect_injection()` checks requests before graph execution
- `guard_request()` blocks unsafe inputs before they reach the supervisor

### 5. Session Audit Log with Cost Tracking
- `SessionAuditLog` tracks:
  - `session_id`
  - `events`
  - `total_cost_usd`
- Session logs are persisted to:
  - `audit_log.jsonl`
- Cost is approximated using mocked token counts for assignment purposes

### 6. Demonstration via `main()`
The `main()` function demonstrates:
- one **orders** request
- one **subscription** request
- printed route and agent used for each run
- final synthesized response for each run
- total cost in USD
- persisted audit log output

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

`.env` must **not** be committed to GitHub.

## How to Run

```bash
python app.py
```

## Example Behavior

When you run the app, it demonstrates:
- an **orders**-type request routed to `orders_agent`
- a **subscription**-type request routed to `subscription_agent`
- route, agent, and synthesized final response printed to stdout
- total approximate cost printed at the end

## Notes

- The project uses **LangGraph** for orchestration
- The supervisor prompt is managed as YAML
- Specialist implementations are intentionally simple for grading
- Token usage and cost are approximated for this assignment

## Submission

Assignment ID: **DAY4 – Multi-Agent Collaboration – Supervisor + Specialists**