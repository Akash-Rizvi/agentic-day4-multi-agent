from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

@dataclass
class SessionAuditLog:
    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(self, agent: str, action: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }


def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")
