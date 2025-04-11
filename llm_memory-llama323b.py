#from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from langchain_ollama import OllamaLLM

# === LLM SETUP ===
llm = OllamaLLM(model="llama3.2:3b")  # Light + fast model for local use

# === VECTOR DB SETUP ===
#qdrant = QdrantClient(":memory:")  # or host URL for persistent store (This means everything is stored in RAM only — it's a temporary, in-memory instance, and all data will be lost when the script exits.)
qdrant = QdrantClient(path="memory_qdrant_db")  # Store all vectors and metadata persistently, pick up from where you left off next time

collection_name = "memory"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if not qdrant.collection_exists(collection_name=collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# qdrant.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
# )

def store_memory(text, person_id: str):
    vector = embedder.encode(text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"text": text, "person_id": person_id}
    )
    qdrant.upsert(collection_name=collection_name, points=[point])

def search_memory(query: str, person_id: str, top_k=3):
    vector = embedder.encode(query).tolist()
    results = qdrant.query_points(collection_name=collection_name,vector=vector,limit=top_k,with_payload=True)
    # Filter by person_id
    filtered = [
        hit.payload["text"] for hit in results if hit.payload.get("person_id") == person_id
    ]
    return filtered


# === MEMORY FUNCTIONS ===
# def store_memory(text):
#     vector = embedder.encode(text).tolist()
#     point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text})
#     qdrant.upsert(collection_name=collection_name, points=[point])

# def search_memory(query, top_k=3):
#     vector = embedder.encode(query).tolist()
#     response = qdrant.search(
#         collection_name=collection_name,
#         query_vector=vector,
#         limit=top_k
#     )
#     return [hit.payload["text"] for hit in response]


# === MAIN CHAT LOOP ===
print("💬 Agent ready. Type 'exit' to quit.\n")
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 🔍 Search memory and inject relevant context
    memory_context = "\n".join(search_memory(user_input))
    enriched_prompt = f"{memory_context}\nUser: {user_input}" if memory_context else user_input

    # 🤖 Get LLM response
    response = llm.invoke(enriched_prompt)
    print(f"Assistant: {response}\n")

    # 💾 Store both sides of conversation in vector memory
    store_memory(f"User: {user_input}")
    store_memory(f"Assistant: {response}")
