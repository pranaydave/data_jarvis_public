# Data Analyst Agent

An **Agentic AI** agent that analyzes your data using OpenAI Assistants API with Code Interpreter. Upload a CSV, ask questions in natural language, and watch agent write code, execute it, and explain insights.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User: "What's driving our sales decline?"                     │
│        ↓                                                        │
│   ┌─────────────────────────────────────────┐                  │
│   │         OpenAI Assistant                 │                  │
│   │         (with Code Interpreter)          │                  │
│   └─────────────────────────────────────────┘                  │
│        ↓                                                        │
│   Agent Plans: "I need to analyze trends, compare periods..."   │
│        ↓                                                        │
│   ┌─────────────────────────────────────────┐                  │
│   │         Code Interpreter                 │                  │
│   │         (Python sandbox)                 │                  │
│   │                                          │                  │
│   │   import pandas as pd                    │                  │
│   │   df = pd.read_csv('data.csv')          │                  │
│   │   # ... analysis code                    │                  │
│   │   plt.savefig('chart.png')              │                  │
│   └─────────────────────────────────────────┘                  │
│        ↓                                                        │
│   Agent Reviews: Output correct? Need more analysis?            │
│        ↓                                                        │
│   Returns: Code + Charts + Explanation                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Natural Language** | Ask questions in plain English |
| **Code Execution** | Agent writes and runs Python code |
| **Visualizations** | Auto-generates charts with dark theme |
| **Iteration** | Agent debugs and retries if needed |
| **Transparency** | See all code that was executed |
| **Multi-turn** | Follow-up questions remember context |


