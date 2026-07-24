# Agent/framework extension summary

Complete mode means the full weak-origin page is already in context. Agentic mode means the origin must be actively retrieved.

| framework / mode | n | trace | flagged | solid | void | reachable | search tool called | origin actually retrieved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| haystack complete n=100 gpt-4o-mini | 100 | 99/100 | 20/100 | 96/96 | - | - | - | - |
| langchain complete n=100 gpt-4o-mini | 100 | 98/100 | 14/100 | 96/96 | - | - | - | - |
| llamaindex complete n=100 gpt-4o-mini | 100 | 97/100 | 5/100 | 96/96 | - | - | - | - |
| gpt-researcher agentic n=30 claude-haiku-4-5 | 30 | - | - | - | 4/30 | 0/30 | void 0/30; reach 0/30 | void 0/30; reach 0/30 |
| gpt-researcher agentic n=30 gpt-4o-mini | 30 | - | - | - | 30/30 | 30/30 | void 0/30; reach 0/30 | void 0/30; reach 0/30 |
| haystack agentic n=30 gpt-4o-mini | 30 | - | - | - | 26/30 | 24/30 | void 29/30; reach 29/30 | void 0/30; reach 3/30 |
| haystack complete n=30 claude-haiku-4-5 | 30 | 8/30 | 0/30 | 30/30 | - | - | - | - |
| haystack complete n=30 gpt-4o-mini | 30 | 30/30 | 6/30 | 30/30 | - | - | - | - |
| langchain complete n=30 claude-opus-4-8 | 30 | 0/30 | 0/30 | 30/30 | - | - | - | - |
| langchain complete n=30 gpt-4o-mini | 30 | 29/30 | 4/30 | 30/30 | - | - | - | - |
| llamaindex agentic n=30 gpt-4o-mini | 30 | - | - | - | 30/30 | 30/30 | void 29/30; reach 29/30 | void 0/30; reach 0/30 |
| llamaindex complete n=30 claude-haiku-4-5 | 30 | 3/30 | 0/30 | 30/30 | - | - | - | - |
| llamaindex complete n=30 gpt-4o-mini | 30 | 30/30 | 2/30 | 30/30 | - | - | - | - |
