# src/query/memory.py
"""
Conversation Memory — stores last 3 (query, response) pairs per session.

Why only 3 turns?
- Gemini Flash free tier has a 1M token/day limit
- Injecting full history into every prompt burns tokens fast
- 3 turns covers 99% of real follow-up scenarios:
    "how do I charge?" → "how long does it take?" → "what about fast charging?"
- Beyond 3 turns, the conversation context is usually stale anyway

Design:
- Plain Python dict — no database, no Redis, no overhead
- Session-scoped: each Streamlit session or API call gets its own key
- Thread-safe for single-process FastAPI (uvicorn default)
- Reset by deleting the session key
"""

from collections import deque
from dataclasses import dataclass, field
from loguru import logger


# Maximum number of (query, response) pairs to keep per session
MAX_TURNS = 3


@dataclass
class Turn:
    """A single conversation turn."""
    query:    str
    response: str


class ConversationMemory:
    """
    In-memory conversation store.
    Instantiate once at app startup and pass around.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        # dict[session_id → deque[Turn]]
        # deque with maxlen automatically evicts oldest turn when full
        self._sessions: dict[str, deque[Turn]] = {}

    def _get_session(self, session_id: str) -> deque[Turn]:
        """Return the deque for this session, creating it if needed."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, query: str, response: str) -> None:
        """
        Add a completed (query, response) turn to memory.
        If memory is full, the oldest turn is automatically dropped.
        """
        session = self._get_session(session_id)
        session.append(Turn(query=query, response=response))
        logger.debug(
            f"Memory [{session_id}]: {len(session)}/{self.max_turns} turns stored"
        )

    def get_turns(self, session_id: str) -> list[Turn]:
        """Return all stored turns for this session, oldest first."""
        return list(self._get_session(session_id))

    def get_context(self, session_id: str) -> str:
        """
        Format stored turns as a context string for prompt injection.

        Output format (injected into the generation prompt):
            User: how do I charge my scooter?
            Assistant: According to your Ather manual, page 12...

            User: how long does it take?
            Assistant: A full charge takes approximately...

        Returns empty string if no history exists (first turn of session).
        """
        turns = self.get_turns(session_id)

        if not turns:
            return ""

        lines = []
        for turn in turns:
            lines.append(f"User: {turn.query}")
            # Truncate long responses in the memory context to save tokens
            # Full response is already stored; this is just for prompt context
            response_preview = (
                turn.response[:300] + "..."
                if len(turn.response) > 300
                else turn.response
            )
            lines.append(f"Assistant: {response_preview}")
            lines.append("")  # blank line between turns

        return "\n".join(lines).strip()

    def get_last_query(self, session_id: str) -> str | None:
        """Return the most recent query, or None if no history."""
        turns = self.get_turns(session_id)
        return turns[-1].query if turns else None

    def reset(self, session_id: str) -> None:
        """Clear all memory for a session. Call on new chat start."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Memory [{session_id}]: reset")

    def session_count(self) -> int:
        """Number of active sessions in memory."""
        return len(self._sessions)


# ── Module-level singleton ────────────────────────────────────────────────────
_memory: ConversationMemory | None = None

def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mem = ConversationMemory(max_turns=3)
    sid = "test-session-1"

    mem.add_turn(sid, "how do I charge?", "Plug in the charger at page 12.")
    mem.add_turn(sid, "how long does it take?", "About 5 hours for a full charge.")
    mem.add_turn(sid, "what about fast charging?", "Fast charge takes 1.5 hours.")

    print("=== Context after 3 turns ===")
    print(mem.get_context(sid))

    # Adding a 4th turn should evict the oldest
    mem.add_turn(sid, "is fast charging safe?", "Yes, it is safe per page 15.")

    print("\n=== Context after 4th turn (oldest evicted) ===")
    print(mem.get_context(sid))

    # Test reset
    mem.reset(sid)
    print(f"\nAfter reset: '{mem.get_context(sid)}'")  # should be empty string