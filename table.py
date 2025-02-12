import mysql.connector
from mysql.connector import Error

# Database connection parameters
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'port': '3306',
    'database': 'characternode'
}

# try:
#     # Establish a connection to the database
#     connection = mysql.connector.connect(**db_config)

#     if connection.is_connected():
#         cursor = connection.cursor()

#         # SQL command to create the table
#         create_table_query = '''
#         CREATE TABLE IF NOT EXISTS user_prompts (
#         user_id INT,
#         Character_name TEXT,
#         Gender TEXT,
#         Backstory TEXT,
#         Greeting TEXT,
#         Character_Type TEXT,
#         voice_name VARCHAR(255),
#         character_id VARCHAR(255)
#         )
#         '''

#         # Execute the SQL command
#         cursor.execute(create_table_query)
#         print("Table created successfully")

# except Error as e:
#     print(f"Error: {e}")

# finally:
#     if connection.is_connected():
#         cursor.close()
#         connection.close()
#         print("MySQL connection is closed")




try:
    # Establish a connection to the database
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        cursor = connection.cursor()

        # SQL command to create the table
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS user_ai_message_info (
        user_id INT,
        character_id VARCHAR(255),
        user_message_id VARCHAR(255),
        ai_message_id VARCHAR(255),
        user_message TEXT,
        ai_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        '''

        # Execute the SQL command
        cursor.execute(create_table_query)
        print("Table created successfully")

except Error as e:
    print(f"Error: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")