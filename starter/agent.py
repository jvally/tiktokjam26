"""Organizer entry point for the enhanced five-workstream agent.

The released starter kept every concern in this file. The required ``Agent`` name
remains here, while its implementation is split into conversation, retrieval,
ranking, policy, and integration modules for independent team ownership.
"""

from shopping_agent import Agent

__all__ = ["Agent"]
