from config import *
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
import wikipedia
import json
from typing import TypedDict

class AgentState(TypedDict):
    user_input: str
    tool_decision: str
    tool_output: str
    final_answer: str

class MistralLangGraphAgent:
    def __init__(self):
        """Initialize Mistral Orchestrator with available tools and LLM"""
        self.hf_token = HF_API_KEY
        self.llm = HuggingFacePipeline.from_model_id(
            # model_id="gpt2",
            model_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            device=-1,  # Use CPU
            pipeline_kwargs={"temperature": 0.3, "max_new_tokens": 512}
        )
        self.tools = self._load_tools()
        self.graph = self._build_graph()

    # ---------------------------
    # Define all tools
    # ---------------------------
    def _load_tools(self):
        def weather_tool(city: str) -> str:
            """Mock Weather API"""
            return f"The weather in {city} is 28°C with clear skies."

        def calculator_tool(expression: str) -> str:
            try:
                return f"Result: {eval(expression)}"
            except Exception as e:
                return f"Error evaluating: {e}"

        def wiki_tool(query: str) -> str:
            try:
                return wikipedia.summary(query, sentences=2)
            except Exception:
                return "No Wikipedia results found."

        return [
            tool(weather_tool, description="Get weather by city name"),
            tool(calculator_tool, description="Evaluate math expressions"),
            tool(wiki_tool, description="Get short Wikipedia summaries")
        ]

    # ---------------------------
    # Build LangGraph orchestration
    # ---------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        def reasoning_step(state):
            question = state["user_input"]
            prompt = f"""
            You are an intelligent AI agent.
            Available tools: Weather, Calculator, Wikipedia.
            User query: {question}
            Decide which tool to use and what input to give, or if no tool is needed, respond with {{"tool": "None"}}.
            Respond strictly in JSON wrapped in ```json.
            Examples:
            For tool use: ```json
            {{"tool": "Calculator", "input": "2+2"}}
            ```
            For no tool: ```json
            {{"tool": "None"}}
            ```
            """
            response = self.llm.invoke(prompt)
            state["tool_decision"] = response
            return state

        def tool_execution_step(state):
            try:
                content = state["tool_decision"].split("```json")[-1].split("```")[0].strip()
                decision = json.loads(content)
                tool_name = decision.get("tool")
                if tool_name and tool_name.lower() != "none":
                    tool_input = decision.get("input")
                    tool = next((t for t in self.tools if t.name.lower() == tool_name.lower()), None)
                    if not tool:
                        state["tool_output"] = "Invalid tool name."
                    else:
                        state["tool_output"] = tool.func(tool_input)
                else:
                    state["tool_output"] = None  # No tool to execute
            except Exception as e:
                state["tool_output"] = f"Failed to parse tool decision: {e}"
            return state

        def summarize_step(state):
            if "tool_output" in state and state["tool_output"] and not state["tool_output"].startswith("Failed to parse"):
                summary_prompt = f"""
                User asked: {state['user_input']}
                Tool output: {state['tool_output']}
                Now give a concise helpful final answer for the user.
                """
                summary = self.llm.invoke(summary_prompt)
            else:
                # Direct answer if no tool was used
                summary_prompt = f"""
                User asked: {state['user_input']}
                Provide a direct, helpful answer.
                """
                summary = self.llm.invoke(summary_prompt)
            state["final_answer"] = summary
            return state

        # Add nodes to graph
        graph.add_node("reasoning", reasoning_step)
        graph.add_node("execute_tool", tool_execution_step)
        graph.add_node("summarize", summarize_step)

        # Define flow with conditional edge
        def route_after_reasoning(state):
            try:
                content = state["tool_decision"].split("```json")[-1].split("```")[0].strip()
                decision = json.loads(content)
                tool_name = decision.get("tool")
                if tool_name and tool_name.lower() != "none":
                    return "execute_tool"
                else:
                    return "summarize"
            except:
                return "summarize"  # Default to summarize on error

        graph.add_conditional_edges("reasoning", route_after_reasoning)
        graph.add_edge("execute_tool", "summarize")
        graph.add_edge("summarize", END)
        graph.set_entry_point("reasoning")

        return graph.compile()

    # ---------------------------
    # Public method: Run query
    # ---------------------------
    def run(self, query: str) -> str:
        """Invoke the orchestration graph on a user query"""
        result = self.graph.invoke({"user_input": query})
        return result["final_answer"]

    # ---------------------------
    # Optional: Add new tools dynamically
    # ---------------------------
    def add_tool(self, name: str, func, description: str):
        """Register a new tool dynamically"""
        self.tools.append(tool(name=name, func=func, description=description))


