from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from uuid import uuid4

# Connect to Cassandra
from cassandra.cluster import Cluster

try:
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()
    print("Connected to Cassandra")
except Exception as e:
    print(f"Unable to connect to Cassandra: {e}")

# Create keyspace if it doesn't exist
# session.execute("""
# CREATE KEYSPACE IF NOT EXISTS tfm 
# WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
# """)

# # Use the keyspace
# session.execute("USE tfm_images;")

# # Create table if it doesn't exist
# session.execute("""
# CREATE TABLE IF NOT EXISTS images (
#     id UUID PRIMARY KEY,
#     name TEXT,
#     description TEXT,
#     image BLOB
# )
# """)

# Function to insert image
def insert_image(name, description, image_path):
    with open(image_path, 'rb') as file:
        image_content = file.read()

    image_id = uuid4()
    query = SimpleStatement("""
        INSERT INTO images (id, name, description, image)
        VALUES (%s, %s, %s, %s)
    """)
    session.execute(query, (image_id, name, description, image_content))
    print(f"Image {name} inserted with ID: {image_id}")

# Insert an image
insert_image("sample_image", "This is a sample image", "test_images/dogs.jpg")
