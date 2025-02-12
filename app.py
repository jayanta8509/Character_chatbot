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
from dotenv import load_dotenv
import database
from werkzeug.exceptions import BadRequest
from asyncio import TimeoutError
from langchain_ollama import ChatOllama
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

quart_app = Quart(__name__)
quart_app = cors(quart_app)

# app = Quart(__name__)
# app = cors(app)
# model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

model = ChatOllama(
    model="dolphin-phi",
    temperature=0.7,
    # other params...
)



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    Character_name: str
    Gender: str
    Backstory: str
    Greeting: str
    Character_Type: str

# "You are a helpful assistant. Create character profiles based on the following attributes: {Character_name}, {Gender}, {Backstory}, {Greeting}, {Character_Type}. Answer all questions to the best of your ability in {language}."

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are now embodying a unique character. Your responses should reflect the personality and background of this character based on the following attributes:
        {Character_name}: This is your name. Use it when introducing yourself or when it's natural in conversation.
        {Gender}: Your gender identity. Let this influence your perspective and experiences subtly.
        {Backstory}: Your personal history. Draw from this to inform your opinions, knowledge, and reactions.
        {Greeting}: Your signature way of saying hello. Use this or variations of it when appropriate.
        {Character_Type}: The archetype or role you represent (e.g., wizard, detective, alien). This should shape your knowledge base and way of thinking.
        {language}: The language you will communicate in. Ensure all responses are in this specified language.
        When asked questions or engaged in conversation, respond as this character would. Incorporate elements of your backstory, use your unique greeting, and let your character type influence your knowledge and perspective. Maintain consistency in your personality and background throughout the interaction. If asked about topics outside your character's expertise, respond as they would - perhaps with uncertainty, curiosity, or by relating it to something they do know. Stay true to your character's voice and experiences while being helpful and engaging."""),
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


@quart_app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = await request.get_json()
        # print(f"Received data: {data}")
        
        if not data:
            raise BadRequest("No JSON data provided")

        query = data.get('query')
        if not query:
            raise BadRequest("Missing required field: query")

        language = data.get('language', 'English')
        thread_id = data.get('character_id')
        if not thread_id:
            raise BadRequest("Missing required field: character_id")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("Missing required field: user_id")

        character_info = await database.get_character_info(thread_id)
        if not character_info:
            raise BadRequest(f"No character found with id: {thread_id}")
        # print(f"Character info: {character_info}")
        
        required_fields = ['character_name', 'gender', 'background_story', 'character_greeting', 'type']
        for field in required_fields:
            if field not in character_info:
                raise BadRequest(f"Missing required field in character_info: {field}")

        character_type = "Anime" if character_info['type'] == 1 else "Photoreal"

        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(query)]

        # print(f"Invoking app with parameters: {input_messages}, {language}, {character_info['character_name']}, {character_info['gender']}, {character_info['background_story']}, {character_info['character_greeting']}, {character_type}")

        try:
            output = await app.ainvoke(
                {
                    "messages": input_messages, 
                    "language": language,
                    "Character_name": character_info['character_name'],
                    "Gender": character_info['gender'],
                    "Backstory": character_info['background_story'],
                    "Greeting": character_info['character_greeting'],
                    "Character_Type": character_type
                },
                config
            )
        except TimeoutError:
            return jsonify({"error": "Request timed out", "status": "error"}), 504
        except Exception as e:
            # print(f"Detailed error in app.ainvoke(): {str(e)}")
            return jsonify({"error": f"Error generating response: {str(e)}", "status": "error"}), 500

        response = output["messages"][-1].content
        # print(f"Generated response: {response}")
        
        message_store = await database.insert_user_ai_message(user_id, thread_id, query, response)

        return jsonify({
            "response": response, 
            "AI_message_id": message_store,
            "status": "success",
            "status_type": 200
        }), 200

    except BadRequest as e:
        # print(f"BadRequest error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        # print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500


from openai import AsyncOpenAI
client = AsyncOpenAI()

@quart_app.route('/tts/<text>')
async def text_to_speech(text):
    async def generate():
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Use a regular for loop instead of async for
        for chunk in response.iter_bytes(chunk_size=4096):
            yield chunk

    return Response(generate(), mimetype="audio/mpeg")


if __name__ == '__main__':
    quart_app.run(debug=True)