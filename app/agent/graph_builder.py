"""LangGraph orchestration graph builder."""

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config.state import SessionState

if TYPE_CHECKING:
    from agent.orchestration import Orchestrator


class GraphBuilder:
    """Builds the LangGraph state graph structure."""

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self.orchestrator = orchestrator

    def build(self) -> CompiledStateGraph:
        memory = MemorySaver()
        graph = StateGraph(SessionState)
        self._add_nodes(graph)
        self._add_edges(graph)
        self._add_conditional_edges(graph)
        return graph.compile(checkpointer=memory)

    def _add_nodes(self, graph: StateGraph) -> None:
        graph.add_node("input_guardrail", self.orchestrator.nodes.input_guardrail.run)
        graph.add_node("agent_node", self.orchestrator.nodes.agent_node.run)
        graph.add_node("tools", ToolNode(self.orchestrator.tools))
        graph.add_node("response", self.orchestrator.nodes.response.run)

    def _add_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, "input_guardrail")
        graph.add_edge("input_guardrail", "agent_node")
        graph.add_edge("tools", "agent_node")
        graph.add_edge("response", END)

    def _add_conditional_edges(self, graph: StateGraph) -> None:
        graph.add_conditional_edges(
            "agent_node", tools_condition, {"tools": "tools", END: "response"}
        )
