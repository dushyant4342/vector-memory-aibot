import csv
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaLLM

# === LLM SETUP ===
llm = OllamaLLM(model="llama3.2:3b")

# === QDRANT SETUP ===
qdrant = QdrantClient(path="memory_qdrant_db")
collection_name = "memory"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if not qdrant.collection_exists(collection_name=collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# === Store memory with person_id ===
# def store_memory(text, person_id: str):
#     vector = embedder.encode(text).tolist()
#     point = PointStruct(
#         id=str(uuid.uuid4()),
#         vector=vector,
#         payload={"text": text, "person_id": person_id}
#     )
#     qdrant.upsert(collection_name=collection_name, points=[point])


def store_memory(text: str, person_id: str, role: str):
    vector = embedder.encode(text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"text": text, "person_id": person_id, "role": role}
    )
    qdrant.upsert(collection_name=collection_name, points=[point])


def search_memory(query: str, person_id: str, top_k=3):
    vector = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True
    )
    print("results:", results)
    return [
        hit.payload["text"]
        for hit in results
        if hit.payload.get("person_id") == person_id  ]

from qdrant_client.http import models


def search_memory(query: str, person_id: str, top_k=20):
    vector = embedder.encode(query).tolist()

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="person_id",
                match=models.MatchValue(value=person_id)
            )
        ]
    )

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

    return [(hit.payload["text"], hit.payload.get("role", "unknown")) for hit in results]



def store_memory_from_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["text"].strip()
            person_id = row["person_id"].strip()
            role = row.get("role", "user").strip().lower()  # Default to 'user' if missing
            if text and person_id and role in ["user", "assistant"]:
                store_memory(text, person_id, role)
        print("✅ CSV memory upsert done.")

# 🧠 Load memory upfront ---------

store_memory_from_csv("user_data_temp.csv")

print("💬 Agent ready. Format: <message>, <person_id>. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Expect input format: <text>,<person_id>
    if "," not in user_input:
        print("⚠️ Please provide input as: <message>, <person_id>\n")
        continue

    # Extract message and person_id
    message_part, person_id = map(str.strip, user_input.rsplit(",", 1))

    # Retrieve relevant memory
    memory_results = search_memory(message_part, person_id=person_id)

    user_msgs = [text for text, role in memory_results if role == "user"][-5:]
    assistant_msgs = [text for text, role in memory_results if role == "assistant"][-5:]


    # De-duplicate and format entries with newlines
    seen = set()
    formatted_memory = []

    # Sort memory results to preserve the original order (optional if Qdrant returns ordered)
    for text, role in memory_results:
        key = f"{role}:{text}"
        if key not in seen:
            seen.add(key)
            formatted_memory.append((role.capitalize(), text))  # ("User", "Message")

    # Interleave into final format
    memory_context = "\n".join(f"{role}: {text}" for role, text in formatted_memory)


    # combined_memory = []
    # for u, a in zip(user_msgs, assistant_msgs):
    #     combined_memory.append(f"User: {u}")
    #     combined_memory.append(f"Assistant: {a}")

    # memory_context = "\n".join(combined_memory)


    print(f"Memory Retrieved ({len(memory_context)}):", memory_context)

    enriched_prompt = f"""You are a professional support assistant. Always respond with empathy, clarity,
    and helpful next steps. Keep the response very short, not more than 2 lines. Avoid asking unnecessary questions.
    Use the provided context to offer practical solutions.{memory_context}User: {message_part}""".strip() if memory_context else message_part

    print(f" 🧠 Enriched Prompt: {enriched_prompt}\n")
    # Get response
    response = llm.invoke(enriched_prompt)
    print(f"Assistant: {response}\n")

    # Store interaction in memory
    store_memory(message_part, person_id=person_id, role="user")
    store_memory(response, person_id=person_id, role="assistant")

    


