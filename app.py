import os
from langchain.chat_models import init_chat_model
from quart import Quart, request, jsonify
from quart import send_from_directory
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
import backstory
import audio
import time
from pathlib import Path
import voice
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

quart_app = Quart(__name__)
quart_app = cors(quart_app, allow_origin="*", allow_headers="*", allow_methods="*")

# ##Local server ollama
# model = ChatOllama(
#     model="dolphin-phi",
#     temperature=0.7,
#     # other params...
# )

##openAI
model = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

## runpoad server

# model = ChatOllama(
#     model="dolphin-mistral:7b",
#     base_url="https://ukeevmnhc9li38-11434.proxy.runpod.net/",
#     temperature=0.7,
#     headers={
#         "Content-Type": "application/json",
#         # Add other headers if needed, for example:
#         # "Authorization": "Bearer YOUR_API_KEY"
#     }
# )


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
    



@quart_app.route('/chat2', methods=['POST'])
async def chat2():
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
        
        character_name = data.get('character_name')
        if not character_name:
            raise BadRequest("Missing required field: character_name")

        gender = data.get('gender')
        if not gender:
            raise BadRequest("Missing required field: gender")


        background_story = data.get('background_story')
        if not background_story:
            raise BadRequest("Missing required field: background_story")

        character_greeting = data.get('character_greeting')
        if not character_greeting:
            raise BadRequest("Missing required field: character_greeting")
        
        character_type = data.get('character_type')
        if not character_type:
            raise BadRequest("Missing required field: character_type")

        
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




@quart_app.route('/create_group', methods=['POST'])
async def create_group():
    try:
        data = await request.get_json()
        group_name = data.get('group_name')
        user_id = data.get('user_id')
        
        if not group_name or not user_id:
            raise BadRequest("Missing required fields: group_name or user_id")

        group_id = await database.insert_create_group(group_name, user_id)
        return jsonify({"group_id": group_id, "status": "success"}), 200

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@quart_app.route('/add_to_group', methods=['POST'])
async def add_to_group():
    try:
        data = await request.get_json()
        group_id = data.get('group_id')
        character_ids = data.get('character_ids')
        
        if not group_id or not character_ids:
            raise BadRequest("Missing required fields: group_id or character_ids")
        
        if not isinstance(character_ids, list):
            raise BadRequest("character_ids must be a list")

        await database.add_characters_to_group(group_id, character_ids)
        return jsonify({"status": "success", "message": f"{len(character_ids)} characters added to the group"}), 200

    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    





##both singale and multi 
# @quart_app.route('/group_chat', methods=['POST'])
# async def group_chat():
#     try:
#         data = await request.get_json()
#         query = data.get('query')
#         language = data.get('language', 'English')
#         group_id = data.get('group_id')
#         user_id = data.get('user_id')

#         if not query or not group_id or not user_id:
#             raise BadRequest("Missing required fields: query, group_id, or user_id")

#         # Extract mentioned character names
#         mentioned_characters = set()
#         words = query.split()
#         actual_query = []
#         for word in words:
#             if word.startswith('@'):
#                 mentioned_characters.add(word[1:].lower())  # Remove '@' and convert to lowercase
#             else:
#                 actual_query.append(word)
        
#         query = ' '.join(actual_query)  # Reconstruct the query without @mentions

#         character_ids = await database.get_group_characters(group_id)
#         responses = []

#         for character_id in character_ids:
#             character_info = await database.get_character_info(character_id)
#             if not character_info:
#                 continue

#             # Check if this character was mentioned or if no characters were mentioned
#             if mentioned_characters and character_info['character_name'].lower() not in mentioned_characters:
#                 continue

#             character_type = "Anime" if character_info['type'] == 1 else "Photoreal"

#             config = {"configurable": {"thread_id": group_id}}
#             input_messages = [HumanMessage(query)]

#             try:
#                 output = await app.ainvoke(
#                     {
#                         "messages": input_messages, 
#                         "language": language,
#                         "Character_name": character_info['character_name'],
#                         "Gender": character_info['gender'],
#                         "Backstory": character_info['background_story'],
#                         "Greeting": character_info['character_greeting'],
#                         "Character_Type": character_type
#                     },
#                     config
#                 )
#                 response = output["messages"][-1].content
#                 message_id = await database.insert_group_ai_message(user_id, group_id, character_id, query, response)
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "response": response,
#                     "AI_message_id": message_id
#                 })

#             except Exception as e:
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "error": str(e)
#                 })

#         return jsonify({
#             "responses": responses, 
#             "status": "success",
#             "status_type": 200
#         }), 200

#     except BadRequest as e:
#         return jsonify({"error": str(e), "status": "error"}), 400
#     except Exception as e:
#         return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500




# @quart_app.route('/group_chat', methods=['POST'])
# async def group_chat():
#     try:
#         data = await request.get_json()
#         query = data.get('query')
#         language = data.get('language', 'English')
#         group_id = data.get('group_id')
#         user_id = data.get('user_id')

#         print(f"Received request: query={query}, group_id={group_id}, user_id={user_id}")

#         if not query or not group_id or not user_id:
#             raise BadRequest("Missing required fields: query, group_id, or user_id")

#         # Extract mentioned character IDs
#         mentioned_character_ids = set()
#         words = query.split()
#         actual_query = []
#         for word in words:
#             if word.startswith('@'):
#                 # Try to convert the mention to an integer ID
#                 try:
#                     character_id = int(word[1:])  # Remove '@' and convert to integer
#                     mentioned_character_ids.add(character_id)
#                     print(f"Found mention for character ID: {character_id}")
#                 except ValueError:
#                     # If it's not a valid ID, just keep it in the query
#                     pass
#             actual_query.append(word)
        
#         query = ' '.join(actual_query)  # Reconstruct the query

#         print(f"Mentioned character IDs: {mentioned_character_ids}")
#         print(f"Reconstructed query: {query}")

#         # Get all characters in the group
#         character_ids = await database.get_group_characters(group_id)
#         print(f"Characters in group {group_id}: {character_ids}")
        
#         # If specific character IDs were mentioned, filter to only those that exist in the group
#         if mentioned_character_ids:
#             filtered_character_ids = []
#             for char_id in character_ids:
#                 if char_id in mentioned_character_ids:
#                     filtered_character_ids.append(char_id)
#         else:
#             filtered_character_ids = character_ids
            
#         print(f"Filtered character IDs to respond: {filtered_character_ids}")
        
#         responses = []

#         for character_id in filtered_character_ids:
#             print(f"Processing character ID: {character_id}")
#             character_info = await database.get_character_info(character_id)
#             if not character_info:
#                 print(f"No info found for character ID: {character_id}")
#                 continue

#             print(f"Character info: {character_info['character_name']}")
#             character_type = "Anime" if character_info['type'] == 1 else "Photoreal"

#             config = {"configurable": {"thread_id": group_id}}
#             input_messages = [HumanMessage(query)]

#             try:
#                 print(f"Sending request to AI for character: {character_info['character_name']}")
#                 output = await app.ainvoke(
#                     {
#                         "messages": input_messages, 
#                         "language": language,
#                         "Character_name": character_info['character_name'],
#                         "Gender": character_info['gender'],
#                         "Backstory": character_info['background_story'],
#                         "Greeting": character_info['character_greeting'],
#                         "Character_Type": character_type
#                     },
#                     config
#                 )
#                 response = output["messages"][-1].content
#                 message_id = await database.insert_group_ai_message(user_id, group_id, character_id, query, response)
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "response": response,
#                     "AI_message_id": message_id
#                 })
#                 print(f"Got response for character {character_info['character_name']}")

#             except Exception as e:
#                 print(f"Error processing character {character_id}: {str(e)}")
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "error": str(e)
#                 })

#         print(f"Final responses: {responses}")
#         return jsonify({
#             "responses": responses, 
#             "status": "success",
#             "status_type": 200
#         }), 200

#     except BadRequest as e:
#         print(f"BadRequest error: {str(e)}")
#         return jsonify({"error": str(e), "status": "error"}), 400
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500




# @quart_app.route('/group_chat', methods=['POST'])
# async def group_chat():
#     try:
#         data = await request.get_json()
#         query = data.get('query')
#         language = data.get('language', 'English')
#         group_id = data.get('group_id')
#         user_id = data.get('user_id')

#         if not query or not group_id or not user_id:
#             raise BadRequest("Missing required fields: query, group_id, or user_id")

#         # Extract mentioned character IDs
#         mentioned_character_ids = set()
#         words = query.split()
#         actual_query = []
#         for word in words:
#             if word.startswith('@'):
#                 # Try to convert the mention to an integer ID
#                 try:
#                     character_id = int(word[1:])  # Remove '@' and convert to integer
#                     mentioned_character_ids.add(character_id)
#                 except ValueError:
#                     # If it's not a valid ID, just keep it in the query
#                     pass
#             actual_query.append(word)
        
#         query = ' '.join(actual_query)  # Reconstruct the query without @mentions

#         # Get all characters in the group
#         character_ids = await database.get_group_characters(group_id)
        
#         # If specific character IDs were mentioned, filter to only those that exist in the group
#         filtered_character_ids = set(character_ids) & mentioned_character_ids if mentioned_character_ids else character_ids
        
#         responses = []

#         for character_id in filtered_character_ids:
#             character_info = await database.get_character_info(character_id)
#             if not character_info:
#                 continue

#             character_type = "Anime" if character_info['type'] == 1 else "Photoreal"

#             config = {"configurable": {"thread_id": character_id}}
#             input_messages = [HumanMessage(query)]

#             try:
#                 output = await app.ainvoke(
#                     {
#                         "messages": input_messages, 
#                         "language": language,
#                         "Character_name": character_info['character_name'],
#                         "Gender": character_info['gender'],
#                         "Backstory": character_info['background_story'],
#                         "Greeting": character_info['character_greeting'],
#                         "Character_Type": character_type
#                     },
#                     config
#                 )
#                 response = output["messages"][-1].content
#                 message_id = await database.insert_group_ai_message(user_id, group_id, character_id, query, response)
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "response": response,
#                     "AI_message_id": message_id
#                 })

#             except Exception as e:
#                 responses.append({
#                     "character_name": character_info['character_name'],
#                     "error": str(e)
#                 })

#         return jsonify({
#             "responses": responses, 
#             "status": "success",
#             "status_type": 200
#         }), 200

#     except BadRequest as e:
#         return jsonify({"error": str(e), "status": "error"}), 400
#     except Exception as e:
#         return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500





@quart_app.route('/group_chat', methods=['POST'])
async def group_chat():
    try:
        data = await request.get_json()
        original_query = data.get('query')
        language = data.get('language', 'English')
        group_id = data.get('group_id')
        user_id = data.get('user_id')

        if not original_query or not group_id or not user_id:
            raise BadRequest("Missing required fields: query, group_id, or user_id")

        # Extract mentioned character IDs and clean the query
        mentioned_character_ids = set()
        words = original_query.split()
        actual_query_words = []
        
        for word in words:
            if word.startswith('@'):
                # Try to convert the mention to an integer ID
                try:
                    character_id = int(word[1:])  # Remove '@' and convert to integer
                    mentioned_character_ids.add(character_id)
                except ValueError:
                    # If it's not a valid ID, keep it in the query
                    actual_query_words.append(word)
            else:
                actual_query_words.append(word)
        
        # The cleaned query without any @mentions
        cleaned_query = ' '.join(actual_query_words)

        # Get all characters in the group
        character_ids = await database.get_group_characters(group_id)
        
        # If specific character IDs were mentioned, filter to only those that exist in the group
        filtered_character_ids = set(character_ids) & mentioned_character_ids if mentioned_character_ids else character_ids
        
        responses = []
        is_first_character = True  # Flag to track the first character

        for character_id in filtered_character_ids:
            character_info = await database.get_character_info(character_id)
            if not character_info:
                continue

            character_type = "Anime" if character_info['type'] == 1 else "Photoreal"

            config = {"configurable": {"thread_id": character_id}}
            # Use the cleaned query for the AI input
            input_messages = [HumanMessage(cleaned_query)]

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
                response = output["messages"][-1].content
                
                # Store the cleaned query only for the first character, use None for others
                stored_query = cleaned_query if is_first_character else None
                message_id = await database.insert_group_ai_message(user_id, group_id, character_id, stored_query, response)
                
                is_first_character = False  # Set flag to false after first character
                
                responses.append({
                    "character_name": character_info['character_name'],
                    "response": response,
                    "AI_message_id": message_id
                })

            except Exception as e:
                responses.append({
                    "character_name": character_info['character_name'],
                    "error": str(e)
                })

        return jsonify({
            "responses": responses, 
            "status": "success",
            "status_type": 200
        }), 200

    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500


@quart_app.route('/backstory', methods=['POST'])
async def backs_tory():
    try:
        data = await request.get_json()
        character_concept = data.get('character_concept')
        if not character_concept:
            raise BadRequest("Missing required field: character_concept")
        response = await backstory.backstory_chat(character_concept)

        return response

    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500



@quart_app.route('/text-to-voice', methods=['POST'])
async def text_to_voice():
    try:
        data = await request.get_json()
        text = data.get('text')
        user_id = data.get('user_id')
        voice_name = data.get('voice_name')
        if not text or not user_id or not voice_name:
            raise BadRequest("Missing required field: text, voice_id or voice_name")
        
        # Generate the audio file
        file_path = await audio.generate_voice(user_id, text, voice_name)
        
        # Get filename from path
        filename = os.path.basename(file_path)
        
        # Create a URL to the audio file with a cache-busting parameter
        base_url = request.host_url.rstrip('/')
        timestamp = int(time.time())  # Import time at the top of your file
        audio_url = f"{base_url}/audio/{filename}?v={timestamp}"
        
        return jsonify({"url": audio_url, "status": "success"})

    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500



@quart_app.route('/voice-to-text-to-voice', methods=['POST'])
async def voice_to_text_to_voice():
    try:
        # Check if there's a file in the request
        if 'audio_file' not in (await request.files):
            raise BadRequest("Missing audio file")

        form = await request.form
        user_id = form.get('user_id')
        voice_name = form.get('voice_name')
        thread_id = form.get('character_id')
        character_name = form.get('character_name')
        gender = form.get('gender')
        background_story = form.get('background_story')
        character_greeting = form.get('character_greeting')
        character_type = form.get('character_type')
        
        if not user_id or not voice_name:
            raise BadRequest("Missing required fields: user_id or voice_name")
        
        # Get the audio file
        audio_file = (await request.files)['audio_file']
        
        # Create a temporary directory if it doesn't exist
        temp_dir = Path(__file__).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Save the uploaded audio file temporarily
        temp_audio_path = temp_dir / f"temp_{user_id}_{int(time.time())}.mp3"
        await audio_file.save(str(temp_audio_path))
        
        # Step 1: Transcribe the audio to text
        transcribed_text = await audio.transcribe_audio(str(temp_audio_path))
        
        if not transcribed_text:
            raise BadRequest("Could not transcribe the audio file")

        # Step 2: Get AI response to the transcribed text
        response_obj = await voice.voice_assistent(
            transcribed_text, 
            thread_id,
            character_name,
            gender,
            background_story,
            character_greeting,
            character_type
        )
        
        # Extract the text response from the JSON response
        if isinstance(response_obj, tuple):
            response_json = await response_obj[0].get_json()
            agent_text = response_json.get('response', '')
        else:
            # Handle the case where response might be in a different format
            agent_text = "I couldn't understand that. Could you try again?"
        
        # Step 3: Generate voice from the AI response
        output_audio_path = await audio.generate_voice(user_id, agent_text, voice_name)
        
        # Clean up the temporary file
        os.remove(str(temp_audio_path))
        
        # Get filename from path
        filename = os.path.basename(output_audio_path)
        
        # Create a URL to the audio file with a cache-busting parameter
        base_url = request.host_url.rstrip('/')
        timestamp = int(time.time())
        audio_url = f"{base_url}/audio/{filename}?v={timestamp}"
        
        return jsonify({
            "url": audio_url,
            "status": "success",
            "status_type": 200
        })
        
    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500


@quart_app.route('/audio/<path:filename>', methods=['GET'])
async def serve_audio(filename):
    response = await send_from_directory('audio', filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@quart_app.route('/', methods=['GET'])
async def base_url():
    return jsonify({
        "status": "running",
        "message": "Character Chatbot API is running",
        "version": "1.0.0"
    }), 200


if __name__ == '__main__':
    quart_app.run(debug=True, port=8056)