import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ Create a database connection to the SQLite database
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"SQLite version: {sqlite3.version}")
        return conn
    except Error as e:
        print(e)

    return conn

def execute_query(conn, sql_query):
    """ Execute a SQL query
    :param conn: Connection object
    :param sql_query: a SQL query
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(sql_query)
    except Error as e:
        print(e)

def create_tables(conn):
    """ Create database tables
    :param conn: Connection object
    :return:
    """
    create_admin_table_query = """
    CREATE TABLE IF NOT EXISTS Admin (
        admin_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE
    );
    """

    create_user_table_query = """
    CREATE TABLE IF NOT EXISTS User (
        user_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE
    );
    """

    create_mri_table_query = """
    CREATE TABLE IF NOT EXISTS MRI (
        mri_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        image_data BLOB,
        FOREIGN KEY (user_id) REFERENCES User (user_id)
    );
    """

    create_cognitive_test_table_query = """
    CREATE TABLE IF NOT EXISTS CognitiveTest (
        test_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        citizenship TEXT,
        dob TEXT,
        favorite_animal TEXT,
        hometown TEXT,
        favorite_color TEXT,
        date_taken TEXT NOT NULL,
        score INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES User (user_id)
    );
    """

    # Create tables
    execute_query(conn, create_admin_table_query)
    execute_query(conn, create_user_table_query)
    execute_query(conn, create_mri_table_query)
    execute_query(conn, create_cognitive_test_table_query)

    conn.commit()

# Create SQLite database connection
conn = create_connection("alzheimer_detection.db")

if conn is not None:
    create_tables(conn)
    conn.close()
else:
    print("Error! Cannot create the database connection.")
