import pinecone as pc

# prompts user to input API key
key = input("Enter API key: ")

# initialize access to database
pc.init(api_key=key, environment="gcp-starter")

# pc.create_index("embeddings", dimension=8, metric="euclidean")

# access 'embeddings' index
index = pc.Index('embeddings')

# inserts five 8D vectors into the embeddings index
# index.upsert(
#     vectors = [
#         {"id": "A", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
#         {"id": "B", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
#         {"id": "C", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
#         {"id": "D", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
#         {"id": "E", "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
#     ]
# )

# finds the vector that best matches the query
matches = index.query(
    vector=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3], # vector query
    top_k=3, # returns top 3 results
    include_values=True
)

# prints the vectors that was the best match
for vector in matches['matches']:
    print(f"Vector: {vector['id']}")
    print(f"Similarity Score: {round(vector['score'], 4)}")
    print(f"Values: {vector['values']}\n")

# deletes the index
# pc.delete_index('embeddings')