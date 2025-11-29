#type: ignore
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelProvider
from typing import cast
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key is None:
    raise Exception("GEMINI_API_KEY is not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=cast(ModelProvider, external_client), # satisfy type checker
    tracing_disabled=True
)