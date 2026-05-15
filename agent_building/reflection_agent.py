"""Reflection (LangGraph): generator produces a draft,
critic reviews it. If approved, ship. Otherwise the critique feeds
back to the generator for another pass. Caps at max_iterations.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI

client = OpenAI(api_key="paste-your-api-key-here")
MAX_ITERS = 3


class State(TypedDict):
    query:     str
    draft:     str
    critique:  str
    iteration: int
    approved:  bool


# ---- Generator: produces or revises the draft ----
def generator_node(state: State) -> dict:
    if state["draft"] and state["critique"]:
        user_msg = (
            f"Original task: {state['query']}\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critic feedback to address:\n{state['critique']}\n\n"
            f"Produce a revised draft."
        )
    else:
        user_msg = state["query"]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a technical writer."},
            {"role": "user",   "content": user_msg},
        ],
    )
    return {
        "draft":     resp.choices[0].message.content,
        "iteration": state["iteration"] + 1,
    }


# ---- Critic: APPROVED or list of changes ----
def critic_node(state: State) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
                "You are a strict editor. Respond 'APPROVED' if the draft "
                "is good, otherwise list specific, actionable changes."},
            {"role": "user", "content": f"Draft:\n{state['draft']}"},
        ],
    )
    critique = resp.choices[0].message.content
    return {
        "critique": critique,
        "approved": critique.strip().upper().startswith("APPROVED"),
    }


# ---- Loop until approved or max_iters ----
def should_continue(state: State) -> str:
    if state["approved"] or state["iteration"] >= MAX_ITERS:
        return END
    return "generator"


# ---- Build the graph: generator -> critic -> (loop back or end) ----
graph = StateGraph(State)
graph.add_node("generator", generator_node)
graph.add_node("critic",    critic_node)
graph.set_entry_point("generator")
graph.add_edge("generator", "critic")
graph.add_conditional_edges(
    "critic", should_continue,
    {"generator": "generator", END: END},
)

app = graph.compile()


# ---- Run it ----
result = app.invoke({
    "query":     "Write an executive summary on AI agent infrastructure.",
    "draft":     "",
    "critique":  "",
    "iteration": 0,
    "approved":  False,
})
print(result["draft"])
