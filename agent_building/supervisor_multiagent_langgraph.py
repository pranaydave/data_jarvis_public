"""Supervisor multi-agent (LangGraph): a coordinator agent
decides at every step which worker to call next, or to FINISH.
Workers append output to the shared message log.

This is where LangGraph earns its weight: state, conditional edges,
and message accumulation get verbose to hand-code.
"""
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI

client = OpenAI(api_key="paste-your-api-key-here")

workers = {
    "researcher": "You are a researcher. Gather facts succinctly.",
    "analyst":    "You are an analyst. Identify patterns and trade-offs.",
    "writer":     "You are a writer. Produce final, polished prose.",
}


class State(TypedDict):
    messages:   List[Dict[str, Any]]
    next:       str
    iterations: int


# ---- Supervisor: decides which worker to run next ----
def supervisor_node(state: State) -> dict:
    history = "\n".join(f"[{m['role']}] {m['content']}" for m in state["messages"])
    decision_prompt = (
        f"Workers: {list(workers.keys())}.\n"
        f"Pick the next worker, or FINISH if done.\n"
        f"Conversation:\n{history}\n\n"
        f'Respond with JSON: {{"next": "<worker_name>" | "FINISH"}}'
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": decision_prompt}],
        response_format={"type": "json_object"},
    )
    decision = json.loads(resp.choices[0].message.content)
    return {"next": decision["next"], "iterations": state["iterations"] + 1}


# ---- Workers: do their part, append to messages, return to supervisor ----
def make_worker(name: str, prompt: str):
    def worker_node(state: State) -> dict:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": str(state["messages"])},
            ],
        )
        new_msg = {"role": name, "content": resp.choices[0].message.content}
        return {"messages": state["messages"] + [new_msg]}
    return worker_node


# ---- Build the graph ----
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
for name, prompt in workers.items():
    graph.add_node(name, make_worker(name, prompt))

# Supervisor branches to a worker, or to END
def route(state: State) -> str:
    return END if state["next"] == "FINISH" else state["next"]

graph.add_conditional_edges(
    "supervisor", route,
    {**{n: n for n in workers}, END: END},
)
# Every worker loops back to the supervisor
for name in workers:
    graph.add_edge(name, "supervisor")

graph.set_entry_point("supervisor")
app = graph.compile()


# ---- Run it ----
result = app.invoke({"messages": [], "next": "", "iterations": 0})
final  = result["messages"][-1]["content"]
print(final)
