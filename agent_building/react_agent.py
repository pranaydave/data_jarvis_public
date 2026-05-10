"""ReAct tool agent: reason -> act (tool call) -> observe -> repeat
until the LLM emits a final answer with no tool calls.

This is OpenAI's native function calling. No LangChain, no LangGraph.
The whole loop fits in one function.
"""
import json
from openai import OpenAI

#Generate your key here platform.openai.com/api-keys
client = OpenAI(api_key="paste-your-api-key-here")


# 1) Tool implementations
def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

def get_weather(location: str) -> str:
    return f"{location}: 72F, sunny"  # mock

TOOL_IMPLS = {"calculator": calculator, "get_weather": get_weather}


# 2) Tool schemas the LLM sees
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]


# 3) The reason -> act -> observe loop
def react_agent(system_prompt: str, query: str, max_iters: int = 6) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
    ]

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # No tool calls -> the LLM is done, this is the final answer
        if not msg.tool_calls:
            return msg.content

        # Otherwise, execute each tool call and feed observations back
        for tc in msg.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments)
            result = TOOL_IMPLS[name](**args)
            messages.append({
                "role":          "tool",
                "tool_call_id":  tc.id,
                "content":       str(result),
            })

    return "(max iterations reached)"


print(react_agent(
    system_prompt="You are a research assistant. Use tools as needed.",
    query="What's the weather in Tokyo, and what's that in Celsius?",
))
