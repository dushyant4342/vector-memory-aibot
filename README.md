# 💬 Local AI Assistant with Long-Term Memory (Qdrant + LangChain + Ollama)

This project demonstrates a **local, memory-enhanced AI assistant** capable of:

- Answering context-aware questions
- Persistently remembering user-specific details
- Working 100% offline using vector search

Built with:
- 🧠 **Qdrant** for storing memory as embeddings
- 🧩 **LangChain** to orchestrate the LLM workflow
- 🤖 **LLaMA 3 (via Ollama)** as the language model
- 🧬 **MiniLM** for fast and effective embeddings

---

## 🔍 What This Assistant Can Do

- 🔁 **Remembers conversations** across sessions
- 💡 **Injects prior knowledge** (e.g. user profile, facts)
- 🧠 **Retrieves top-5 semantically similar memories**
- 👥 Supports **multi-user context** using `person_id`
- 🗃️ Lets you **upload a CSV** to initialize memory

> 📌 **Vector DB is best for semantic search and context injection.**

---

## 🚀 How It Works (Main Code : final_agent_qdrant_memory.py)

1. Upload a CSV with columns: `text`, `person_id`, `role` (e.g., "assistant")
2. These facts get stored in Qdrant and tagged as `type=info`
3. At runtime:
   - You pass `user query, person_id`
   - System searches vector DB for relevant memory using embeddings
   - Retrieves latest 5 chats (user + assistant) and `User Info`
4. LangChain builds a rich prompt and sends it to Ollama
5. Response is generated and added back to memory

---

## 📁 File Structure

```
├── final_agent.py              # Main script
├── user_data_temp.csv          # Initial memory (optional)
├── memory_qdrant_db/           # Persistent vector DB
└── README.md                   # This file
```

---

## 🛠️ Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Make Sure Ollama is Running
```bash
ollama serve
```
Use the model `llama3.2:3b` or similar.

### 3. (Optional) Prepare CSV
```csv
text,person_id,role
"Sridhar is 25 and works in fintech and has an outstanding amount of 1000 Rs due on 15th May.",10000000000000000002,assistant
```
This gets stored as static `User Info:` memory.

### 4. Run the Agent
```bash
python final_agent.py
```
Then type messages like:
```
How much should I pay?, 10000000000000000002
```

---

## 🔧 Configuration Summary

| Feature            | Description                                      |
|--------------------|--------------------------------------------------|
| Vector DB          | Qdrant (local, persistent)                       |
| Embedding Model    | all-MiniLM-L6-v2                                 |
| LLM Backend        | LLaMA 3 (3b) via Ollama                          |
| Memory Strategy    | Last 5 user + assistant turns, plus "info" tags |
| Person Tracking    | Every memory tied to `person_id`                |

---

## 📦 Ideal Use Cases

- Fintech bots with user-specific history
- On-device assistants with memory
- Smart customer support agents
- Context-rich automation for enterprise workflows

---

## 🧪 Example Conversation

```
User Info: Sridhar is 25 and works in fintech and has an outstanding amount of 1000 Rs due on 15th May.
User: How much should I pay?
Assistant: You should aim to pay 1000 Rs by May 15th to avoid any late fees.
```

---

## 🧠 Future Plans

- ✅ Streamlit UI
- ✅ Memory summarization
- ✅ Intent tagging and fallback
- ✅ API version with FastAPI




- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Customer Support: Recall user history to resolve issues faster.
KYC & Onboarding: Track progress across sessions.
Fraud Detection: Remember behavior patterns over time.
Personal Finance Assistants: Offer tailored advice using past interactions.
Loan Advisors: Recall preferences, explain terms based on prior questions.

Memory = context = smarter, faster, more human-like service.


✅ Benefits of storing data with a unique_id in vector db
Clean separation of user contexts
Easier debugging and memory tracing
Prepares your system for multi-user, multi-agent environments

