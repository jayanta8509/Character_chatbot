import os
from langchain.chat_models import init_chat_model
from quart import Quart, request, jsonify
from quart import Quart, Response
from quart_cors import cors
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import database
from werkzeug.exceptions import BadRequest
from asyncio import TimeoutError
from langchain_ollama import ChatOllama
import random


# #Local server ollama
# model = ChatOllama(
#     model="dolphin-phi",
#     temperature=0.7,
#     max_tokens=100,
#     top_k=40,        # Controls diversity by limiting to top 40 token choices
#     top_p=0.9       # Nucleus sampling - only considers tokens with cumulative probability of 0.9
#     # other params...
# )



# ## runpoad server ollama
# model = ChatOllama(
#     model="dolphin-mistral:7b",
#     base_url="https://h3z78w4owudgje-11434.proxy.runpod.net/",
#     temperature=0.7,
#      max_tokens=150,
#     top_k=40,        # Controls diversity by limiting to top 40 token choices
#     top_p=0.9,  
#     headers={
#         "Content-Type": "application/json",
#         # Add other headers if needed, for example:
#         # "Authorization": "Bearer YOUR_API_KEY"
#     }
# )

model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    character_concept: str  # Single parameter for generating backstory

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Describe your {character_concept} in a few sentences. Include details like their name, personality, appearance, abilities (if any), and the world they live in (fantasy, sci-fi, modern, etc.). You can also mention any significant events in their past or their goals. The more details you provide, the richer the backstory will be!"),
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


# @quart_app.route('/generate_backstory', methods=['POST'])
async def backstory_chat(character_concept):
    try:
        if not character_concept:
            raise BadRequest("Missing required field: character_concept")
        
        random_number = (f'{random.randrange(0, 100000):05}')
        print(f"Random number: {random_number}")

        config = {"configurable": {"thread_id": random_number}}
        input_messages = [HumanMessage(content=character_concept)]

        try:
            output = await app.ainvoke(
                {
                    "messages": input_messages,
                    "character_concept": character_concept  # Add this line to include character_concept
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
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500




# if __name__ == "__main__":
#     quart_app.run(debug=True, port=5000)