import os
from langchain.chat_models import init_chat_model
from quart import Quart, request, jsonify
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from werkzeug.exceptions import BadRequest
from asyncio import TimeoutError
from langchain_ollama import ChatOllama

# model = ChatOllama(
#     model="dolphin-mistral:7b",
#     base_url="https://h3z78w4owudgje-11434.proxy.runpod.net/",
#     temperature=0.7,
#     headers={
#         "Content-Type": "application/json",
#     }
# )

model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    Character_name: str
    Gender: str
    Backstory: str
    Greeting: str
    Character_Type: str

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are {Character_name}, a {Gender} {Character_Type} with this background: {Backstory}. 
            When greeting someone, use: {Greeting}
            Always respond in {language}.
            Keep all responses concise and brief (under 3 sentences when possible)."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the workflow
workflow = StateGraph(state_schema=State)

async def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = await model.ainvoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Initialize memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


async def voice_assistent(query, thread_id,character_name,gender,background_story,character_greeting,character_type,language='English',):
    try:
        character_type = "Anime" if character_type == 1 else "Photoreal"

        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(query)]

        try:
            output = await app.ainvoke(
                {
                    "messages": input_messages, 
                    "language": language,
                    "Character_name": character_name,
                    "Gender": gender,
                    "Backstory": background_story,
                    "Greeting": character_greeting,
                    "Character_Type": character_type
                },
                config
            )
        except TimeoutError:
            return jsonify({"error": "Request timed out", "status": "error"}), 504
        except Exception as e:
            return jsonify({"error": f"Error generating response: {str(e)}", "status": "error"}), 500

        response = output["messages"][-1].content

        return jsonify({
            "response": response, 
            "status": "success",
            "status_type": 200
        }), 200

    except BadRequest as e:
        # print(f"BadRequest error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        # print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500