import asyncio
import os
import logging

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import (
    AuthorRole,
    ChatMessageContent,
)


async def process_request(request: str):

    agent = ChatCompletionAgent(
        name="PersonalAssistant",
        instructions="You are a personal assistant that can answer questions and provide information.",
        service=AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        ),
    )

    logging.info(f"Processing request: {request}")

    initial_message = ChatMessageContent(role=AuthorRole.USER, content=request)
    chat_history = [initial_message]

    response_count = 0
    async for message in agent.invoke(chat_history):
        response_count += 1
        logging.info(f"{agent.name}: {message.content}")

    if response_count == 0:
        logging.info("No response from agent")
    else:
        logging.info(f"Got {response_count} response(s) from agent")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(process_request("What is the weather today like in London?"))
