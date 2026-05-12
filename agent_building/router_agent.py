"""Router multi-agent: a classifier picks one specialist
agent, that specialist answers. No looping, no back-and-forth.

The router decision is itself an LLM call returning structured JSON.
"""
import json
from openai import OpenAI

client = OpenAI(api_key="paste-your-api-key-here")


# Specialists, each with a focused system prompt
specialists = {
    "mathematician": "You are a mathematician. Show work step-by-step.",
    "writer":        "You are a creative writer. Polished prose.",
    "coder":         "You are a senior Python engineer. Clean code, brief comments.",
}


def router_agent(router_prompt: str, query: str) -> str:
    # Step 1: classify -> pick a specialist
    routing_system = (
        router_prompt
        + f"\n\nAvailable specialists: {', '.join(specialists.keys())}.\n"
        + 'Respond with JSON: {"specialist": "<name>", "reason": "..."}'
    )
    routing_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": routing_system},
            {"role": "user",   "content": query},
        ],
        response_format={"type": "json_object"},
    )
    decision = json.loads(routing_resp.choices[0].message.content)
    chosen   = decision["specialist"]

    # Step 2: specialist answers
    specialist_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": specialists[chosen]},
            {"role": "user",   "content": query},
        ],
    )
    return specialist_resp.choices[0].message.content


print(router_agent(
    router_prompt="You are a router. Classify and dispatch.",
    query="Write a haiku about gradient descent.",
))
