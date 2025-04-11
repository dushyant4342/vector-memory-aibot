# Updated complete version of the code with "type": "info" tagging for CSV-based memory

import csv
import uuid
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
from langchain_ollama import OllamaLLM

# === LLM SETUP ===
llm = OllamaLLM(model="llama3.2:3b")

# === QDRANT SETUP ===
qdrant = QdrantClient(path="memory_qdrant_db")
collection_name = "memory"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection if not exists
if not qdrant.collection_exists(collection_name=collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# === Store memory with person_id, role, timestamp, and optional type ===
def store_memory(text: str, person_id: str, role: str, memory_type: str = "chat"):
    vector = embedder.encode(text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "text": text,
            "person_id": person_id,
            "role": role,
            "timestamp": time.time(),
            "type": memory_type
        }
    )
    qdrant.upsert(collection_name=collection_name, points=[point])

# === Search memory with filtering by person_id ===
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
    return [
        (
            hit.payload["text"],
            hit.payload.get("role", "unknown"),
            hit.payload.get("timestamp", 0),
            hit.payload.get("type", "chat")
        )
        for hit in results
    ]

# === Load memory from CSV and tag it as type=info ===
def store_memory_from_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["text"].strip()
            person_id = row["person_id"].strip()
            role = row.get("role", "user").strip().lower()
            if text and person_id and role in ["user", "assistant"]:
                store_memory(text, person_id, role, memory_type="info")
        print("✅ CSV memory upsert done.")









# === MAIN LOOP ===
#store_memory_from_csv("user_data_temp.csv")








print("💬 Agent ready. Format: <message>, <person_id>. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    if "," not in user_input:
        print("⚠️ Please provide input as: <message>, <person_id>\n")
        continue

    message_part, person_id = map(str.strip, user_input.rsplit(",", 1))
    memory_results = search_memory(message_part, person_id=person_id)

    # Sort by timestamp
    sorted_memory = sorted(memory_results, key=lambda x: x[2])

    seen = set()
    user_info_lines = []
    chat_lines = []

    for text, role, _, memory_type in sorted_memory:
        key = f"{role}:{text}"
        if key not in seen:
            seen.add(key)
            if memory_type == "info":
                user_info_lines.append(f"User Info: {text}")
            else:
                chat_lines.append(f"{role.capitalize()}: {text}")

    memory_context = "\n".join(user_info_lines + chat_lines[-10:])  # Always keep info, rotate last 10 chat lines

    system_prompt = (
        "You are a professional support assistant. Always respond with empathy, clarity, "
        "and helpful next steps. Keep the response very short, not more than 2 lines. "
        "Avoid asking unnecessary questions. Use the provided context to offer practical solutions."
    )

    enriched_prompt = (
        f"{system_prompt}\n\n{memory_context}\nUser: {message_part}"
        if memory_context else f"{system_prompt}\n\nUser: {message_part}"
    )

    print(f"\n🧠 Enriched Prompt:\n{enriched_prompt}\n")

    response = llm.invoke(enriched_prompt)
    print("="*262)
    print(f" New Assistant Response: {response}\n")
    
    store_memory(message_part, person_id=person_id, role="user")
    store_memory(response, person_id=person_id, role="assistant")
