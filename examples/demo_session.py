from __future__ import annotations

import json

from starter.agent import Agent


def main() -> None:
    agent = Agent("data/catalog.jsonl")
    session_id = "demonstrated_session"
    agent.reset(session_id, {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": ["comfort", "fit", "durability"],
        "summary": "Prior purchases emphasize comfort, fit, and durability.",
    })
    messages = [
        "I'm looking for Women's Shoes Athletic Running, but I'm still exploring.",
        "For that, what matters is: breathable mesh; color: black.",
        "For that, what matters is: lightweight; budget around $80.",
    ]
    for turn, user_message in enumerate(messages, 1):
        response = agent.respond(session_id, user_message, turn, 10)
        print(json.dumps({"turn": turn, "user": user_message, "agent": response}, indent=2))


if __name__ == "__main__":
    main()
