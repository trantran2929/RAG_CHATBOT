from dataclasses import dataclass, field
from typing import List
from langgraph.graph.message import add_messages

@dataclass
class State:
    # messages là một channel, reducer do langgraph cung cấp (add_messages)
    messages: List = field(default_factory=list, metadata={"reducer": add_messages})
