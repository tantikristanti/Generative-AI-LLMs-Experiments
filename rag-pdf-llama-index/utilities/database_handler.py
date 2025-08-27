import psycopg2

def connect_db(db_name,
               db_type="postgres",
               host="localhost",
               user="newuser", 
               password="newuser",
               port="5432"
               ):
    """
    Load the pre-trained embedding model. 
    If the model_name is a filepath on disc, it loads the model from that path. 
    If it is not a path, it first tries to download a pre-trained SentenceTransformer model.

    Args:
        db_name, host, port, user, password (str): The data for creating a vector store

    Returns:
        HuggingFaceEmbedding: The HuggingFace embedding model
    """

    conn = psycopg2.connect(dbname=db_type,
                            host=host,
                            user=user,
                            password=password,
                            port=port,)

    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
    