import mysql.connector
from mysql.connector import Error
import uuid
import asyncio

async def insert_user_prompt(user_id, character_name, gender, backstory, greeting, character_type, voice_name):
    # Auto-generate character_id
    character_id = str(uuid.uuid4())

    # Database connection parameters
    db_config = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'port': '3306',
        'database': 'characternode'
    }

    # SQL query
    query = """
    INSERT INTO user_prompts 
    (user_id, Character_name, Gender, Backstory, Greeting, Character_Type, voice_name, character_id)
    VALUES 
    (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        # Establish a database connection
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()

            # Execute the query with parameters
            cursor.execute(query, (user_id, character_name, gender, backstory, greeting, character_type, voice_name, character_id))

            # Commit the changes
            connection.commit()

            # print(f"Record inserted successfully. Generated character_id: {character_id}")

    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            # print("MySQL connection is closed")

    return character_id



async def get_character_info(character_id):
    db_config = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'port': '3306',
        'database': 'characternode'
    }

    query = """
    SELECT character_name, gender, background_story, character_greeting, type
    FROM characters
    WHERE id = %s
    """

    try:
        # Use a connection pool or run in a thread pool executor for better performance in async context
        connection = await asyncio.to_thread(mysql.connector.connect, **db_config)
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            await asyncio.to_thread(cursor.execute, query, (character_id,))
            result = await asyncio.to_thread(cursor.fetchone)
            
            if result:
                return result
            else:
                return None

    except Error as e:
        print(f"Error: {e}")
        return None

    finally:
        if connection.is_connected():
            await asyncio.to_thread(cursor.close)
            await asyncio.to_thread(connection.close)




async def insert_user_ai_message(user_id, character_id, user_message, ai_message):
    user_message_id = f"User_message_{str(uuid.uuid4())}"
    ai_message_id = f"AI_message_{str(uuid.uuid4())}"

    # Database connection parameters
    db_config = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'port': '3306',
        'database': 'characternode'
    }

    # SQL query
    query = """
    INSERT INTO user_ai_message_info 
    (user_id, character_id, user_message_id, ai_message_id, user_message, ai_message)
    VALUES 
    (%s, %s, %s, %s, %s, %s)
    """

    def db_operation():
        try:
            connection = mysql.connector.connect(**db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute(query, (user_id, character_id, user_message_id, ai_message_id, user_message, ai_message))
                connection.commit()
        except Error as e:
            print(f"Error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    # Run the database operation in a separate thread
    await asyncio.to_thread(db_operation)

    return ai_message_id



