from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env")

model = ChatOpenAI(temperature=0.1, max_tokens=1000, timeout=30)


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


tools = [search]

agent = create_agent(model, tools=tools)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"},
)

print(result)


# (rag-agent) mp@mac rag-agent % uv run ./agent.py
# {
#     "messages": [
#         HumanMessage(
#             content="Explain machine learning",
#             additional_kwargs={},
#             response_metadata={},
#             id="dfcc3aa8-1406-4f8f-b54c-7e0e5ccdb496",
#         ),
#         AIMessage(
#             content="",
#             additional_kwargs={"refusal": None},
#             response_metadata={
#                 "token_usage": {
#                     "completion_tokens": 14,
#                     "prompt_tokens": 45,
#                     "total_tokens": 59,
#                     "completion_tokens_details": {
#                         "accepted_prediction_tokens": 0,
#                         "audio_tokens": 0,
#                         "reasoning_tokens": 0,
#                         "rejected_prediction_tokens": 0,
#                     },
#                     "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
#                 },
#                 "model_provider": "openai",
#                 "model_name": "gpt-3.5-turbo-0125",
#                 "system_fingerprint": None,
#                 "id": "chatcmpl-Cq8wDykose1lRLBVRzseEhvqzIo9f",
#                 "service_tier": "default",
#                 "finish_reason": "tool_calls",
#                 "logprobs": None,
#             },
#             id="lc_run--019b4e31-d715-7212-9fb1-d90b8df53e6b-0",
#             tool_calls=[
#                 {
#                     "name": "search",
#                     "args": {"query": "machine learning"},
#                     "id": "call_MzjjKfBK23fozyR788TXIY3R",
#                     "type": "tool_call",
#                 }
#             ],
#             usage_metadata={
#                 "input_tokens": 45,
#                 "output_tokens": 14,
#                 "total_tokens": 59,
#                 "input_token_details": {"audio": 0, "cache_read": 0},
#                 "output_token_details": {"audio": 0, "reasoning": 0},
#             },
#         ),
#         ToolMessage(
#             content="Results for: machine learning",
#             name="search",
#             id="f1b7dae0-d5ba-44ee-b843-1403c1857982",
#             tool_call_id="call_MzjjKfBK23fozyR788TXIY3R",
#         ),
#         AIMessage(
#             content="Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and models that allow computers to learn and make predictions or decisions based on data. It involves the use of statistical techniques to enable machines to improve their performance on a specific task without being explicitly programmed. Machine learning algorithms can be categorized into supervised learning, unsupervised learning, and reinforcement learning, each serving different purposes in training models. The field of machine learning has applications in various industries, including healthcare, finance, marketing, and more.",
#             additional_kwargs={"refusal": None},
#             response_metadata={
#                 "token_usage": {
#                     "completion_tokens": 103,
#                     "prompt_tokens": 71,
#                     "total_tokens": 174,
#                     "completion_tokens_details": {
#                         "accepted_prediction_tokens": 0,
#                         "audio_tokens": 0,
#                         "reasoning_tokens": 0,
#                         "rejected_prediction_tokens": 0,
#                     },
#                     "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
#                 },
#                 "model_provider": "openai",
#                 "model_name": "gpt-3.5-turbo-0125",
#                 "system_fingerprint": None,
#                 "id": "chatcmpl-Cq8wEmj1zl3QjFLl5koWPzg0hhI6I",
#                 "service_tier": "default",
#                 "finish_reason": "stop",
#                 "logprobs": None,
#             },
#             id="lc_run--019b4e31-dcbe-7e81-ab8b-7c9ef471bac3-0",
#             usage_metadata={
#                 "input_tokens": 71,
#                 "output_tokens": 103,
#                 "total_tokens": 174,
#                 "input_token_details": {"audio": 0, "cache_read": 0},
#                 "output_token_details": {"audio": 0, "reasoning": 0},
#             },
#         ),
#     ]
# }
