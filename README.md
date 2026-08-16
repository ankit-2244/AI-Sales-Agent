# AI Sales Agent

An agentic AI sales assistant designed to automate key stages of the sales workflow, including knowledge-base question answering, lead qualification, CRM integration, and demo booking.

The system combines **LLM reasoning, RAG, hybrid retrieval, reranking, LangGraph-based agent orchestration, and tool execution** behind a FastAPI API.

## Live Demo

**Live API:**
https://ai-sales-agent.fastapicloud.dev/

**Interactive API Documentation:**
https://ai-sales-agent.fastapicloud.dev/docs

The root endpoint confirms that the deployed API is running.

The `/docs` endpoint provides an interactive Swagger UI where the `/chat` endpoint can be tested directly from a browser.

## Web UI (frontend)

The API had no frontend. A chat UI now lives in `frontend/` and calls `POST /chat`.

Run the UI against the live API (no Gemini key):

```bash
cd ~/AI-Sales-Agent
chmod +x run_ui.sh
./run_ui.sh
```

Open **http://localhost:5500** (not `127.0.0.1` if you want the microphone).

If you run the FastAPI app locally, the same UI is at **http://localhost:8000/ui/**.

To put the UI on Netlify: publish directory `frontend`, functions `netlify/functions`. `/chat` and `/health` are proxied to the live API so CORS is not required.

---

# 1. Project Overview

Sales teams spend significant time answering repetitive product questions, qualifying inbound leads, updating CRM systems, and scheduling product demonstrations.

This project implements an AI Sales Agent that can:

* Answer product and solution-related questions using a knowledge base
* Retrieve relevant information using hybrid search
* Use semantic/vector retrieval and BM25 keyword retrieval
* Rerank retrieved information
* Qualify sales leads based on provided information
* Assign a qualification score
* Add qualified leads to a CRM
* Book product demonstrations
* Maintain conversation context using thread IDs
* Expose the complete workflow through a FastAPI REST API

The goal is to demonstrate a practical **agentic AI workflow rather than a simple chatbot**.

---

# 2. Key Capabilities

| Capability           | Description                                              |
| -------------------- | -------------------------------------------------------- |
| Knowledge Q&A        | Answers questions using the sales knowledge base         |
| RAG                  | Retrieves relevant knowledge before generating an answer |
| Hybrid Search        | Combines semantic vector search with BM25 keyword search |
| Reranking            | Improves ordering of retrieved documents                 |
| Lead Qualification   | Extracts lead information and evaluates qualification    |
| CRM Integration      | Creates a CRM lead and returns a CRM ID                  |
| Demo Booking         | Books a product demonstration and returns a meeting ID   |
| Conversation Threads | Supports conversations using `thread_id`                 |
| REST API             | FastAPI endpoint for application integration             |
| Cloud Deployment     | Deployed as a publicly accessible API                    |

---

# 3. Architecture

```text
                         User
                           |
                           v
                    FastAPI /chat
                           |
                           v
                    AI Sales Agent
                           |
                           v
                    LangGraph Workflow
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        Knowledge      Qualification   Tool Actions
        Retrieval          |             |
             |             |       +-----+------+
             v             v       |            |
       Hybrid Search   Lead Score  CRM      Demo Booking
             |
       +-----+------+
       |            |
       v            v
   Vector Search   BM25
       |            |
       +-----+------+
             |
             v
          Reranking
             |
             v
      Relevant Context
             |
             v
           LLM
             |
             v
        Final Answer
```

---

# 4. Agent Workflow

The agent follows a state-driven workflow.

A typical request passes through the following stages:

```text
User Request
     |
     v
Intent / Agent Processing
     |
     +------------------+
     |                  |
     v                  v
Knowledge Query      Sales Action
     |                  |
     v                  +-------------------+
Hybrid Retrieval        |         |         |
     |                   v         v         v
     v                  CRM     Demo     Qualification
Reranking
     |
     v
LLM Response
     |
     v
User
```

The workflow is implemented using LangGraph so that the application can route requests through different processing and tool-execution paths.

---

# 5. Retrieval-Augmented Generation

The knowledge-answering component uses a RAG pipeline.

The process is:

1. User submits a question.
2. The system generates an embedding for the query.
3. Vector search retrieves semantically similar knowledge.
4. BM25 retrieves keyword-based matches.
5. Results from both retrieval methods are combined using Reciprocal Rank Fusion.
6. Retrieved information is reranked.
7. Relevant context is passed to the agent.
8. The LLM generates the final response.

This reduces dependence on the model's parametric knowledge and grounds responses in the project's sales knowledge base.

---

# 6. Hybrid Search

The project combines two retrieval approaches.

### Semantic Vector Search

Semantic search retrieves documents based on meaning.

The knowledge base is embedded and stored in ChromaDB.

```text
User Query
    |
    v
Embedding
    |
    v
ChromaDB
    |
    v
Semantic Results
```

### BM25 Search

BM25 provides keyword-based retrieval.

This is useful when the query contains important exact terms such as:

* CRM
* lead qualification
* demo
* sales representatives
* pricing

### Reciprocal Rank Fusion

The project combines vector-search ranking and BM25 ranking using Reciprocal Rank Fusion.

This provides a balance between:

* semantic similarity
* exact keyword matching

---

# 7. Reranking

After the initial retrieval stage, the retrieved documents are reranked before being supplied to the agent.

The purpose of reranking is to improve the relevance of the final context.

The retrieval pipeline is therefore:

```text
Query
  |
  v
Vector Search + BM25
  |
  v
Candidate Documents
  |
  v
RRF / Hybrid Ranking
  |
  v
Reranking
  |
  v
Top Relevant Context
```

---

# 8. Lead Qualification

The agent can qualify inbound sales leads.

Example request:

```text
Please qualify this lead. My name is Ankit.
Company: Acme Technologies.
We need AI sales automation.
Our budget is $50,000 per year.
```

Example response:

```json
{
  "answer": "Thanks for the information. Your lead has been classified as 'qualified' with a qualification score of 100."
}
```

The workflow demonstrates structured lead qualification rather than simply returning a generic conversational response.

---

# 9. CRM Integration

The agent can add a lead to the CRM.

Example request:

```text
Please add me to the CRM. My name is Ankit,
my email is ankit@example.com,
and I work at Acme Technologies.
```

Example response:

```json
{
  "answer": "Your lead has been added to our CRM. CRM ID: CRM-ANKIT-ACME-TECHNOLOGIES."
}
```

The CRM tool demonstrates how an agent can move from natural-language understanding to an external business action.

---

# 10. Demo Booking

The agent can also schedule product demonstrations.

Example request:

```text
I want to book a demo. My name is Ankit,
my email is ankit@example.com,
and I prefer tomorrow at 11 AM.
```

Example response:

```json
{
  "answer": "Your demo has been booked successfully. Your meeting ID is DEMO-ANKIT-001."
}
```

---

# 11. API

The application exposes a REST API using FastAPI.

## Endpoint

```text
POST /chat
```

## Request

```json
{
  "message": "Does the agent work with a CRM?",
  "thread_id": "demo-test-1"
}
```

## Response

```json
{
  "answer": "Yes, the agent synchronizes leads to a CRM."
}
```

---

# 12. Interactive API Documentation

Swagger UI is available at:

```text
https://ai-sales-agent.fastapicloud.dev/docs
```

The `/chat` endpoint can be executed directly from the browser.

This makes the deployment easy to demonstrate without requiring the interviewer to install the project locally.

---

# 13. Example API Tests

### Test 1 — Knowledge Retrieval

Request:

```json
{
  "message": "Does the agent work with a CRM?",
  "thread_id": "cloud-test-1"
}
```

Expected behavior:

```text
The agent explains that it can synchronize leads with a CRM.
```

---

### Test 2 — Lead Qualification

Request:

```json
{
  "message": "Please qualify this lead. My name is Ankit. Company: Acme Technologies. We need AI sales automation. Our budget is $50,000 per year.",
  "thread_id": "final-qualification-test-1"
}
```

Expected behavior:

```text
The lead is classified as qualified and a qualification score is returned.
```

---

### Test 3 — CRM

Request:

```json
{
  "message": "Please add me to the CRM. My name is Ankit, my email is ankit@example.com, and I work at Acme Technologies.",
  "thread_id": "final-crm-test-1"
}
```

Expected behavior:

```text
A CRM ID is returned after the lead is added.
```

---

### Test 4 — Demo Booking

Request:

```json
{
  "message": "I want to book a demo. My name is Ankit, my email is ankit@example.com, and I prefer tomorrow at 11 AM.",
  "thread_id": "final-demo-test-1"
}
```

Expected behavior:

```text
A meeting ID is returned after the demo is booked.
```

---

# 14. Project Structure

```text
AI-Sales-Agent/
│
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── api.py
│   ├── graph.py
│   ├── hybrid_search.py
│   ├── index_knowledge.py
│   ├── reranker.py
│   ├── search.py
│   └── tools.py
│
├── data/
│   └── knowledge_base.txt
│
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
├── embedding_test.py
└── semantic_search.py
```

---

# 15. Technology Stack

### Programming Language

* Python

### LLM

* Google Gemini

### Agent Orchestration

* LangGraph

### API

* FastAPI
* Uvicorn

### Retrieval

* ChromaDB
* BM25
* Hybrid Retrieval
* Reciprocal Rank Fusion
* Reranking

### Environment Management

* python-dotenv

### Version Control

* Git
* GitHub

### Deployment

* FastAPI Cloud

---

# 16. Local Setup

Clone the repository:

```bash
git clone https://github.com/ankit-2244/AI-Sales-Agent.git
cd AI-Sales-Agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 17. Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is intentionally excluded from Git using `.gitignore`.

Never commit API keys or other credentials to the repository.

---

# 18. Running Locally

Start the FastAPI application:

```bash
uvicorn app.api:app --reload
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 19. Knowledge Base Indexing

The knowledge base is stored in:

```text
data/knowledge_base.txt
```

The indexing script creates embeddings and stores them in ChromaDB.

Run:

```bash
python -m app.index_knowledge
```

The resulting local ChromaDB directory is excluded from Git because it is a generated local database.

---

# 20. Security

The project follows basic credential-management practices.

* API keys are stored in environment variables.
* `.env` is excluded from Git.
* Local ChromaDB files are excluded from Git.
* Python cache files are excluded from Git.
* Secrets are not included in the repository.

---

# 21. Testing

The system was tested through the deployed FastAPI endpoint for:

* Knowledge-base question answering
* CRM-related questions
* Lead qualification
* CRM lead creation
* Demo booking

The deployed API returned HTTP `200` responses for the successful test cases.

---

# 22. Limitations

This project is a prototype/MVP designed to demonstrate an agentic AI sales workflow.

Potential production improvements include:

* Real CRM API integration such as Salesforce or HubSpot
* Real calendar integration such as Google Calendar
* Authentication and authorization
* Persistent production-grade conversation storage
* Observability and tracing
* Rate limiting
* Automated evaluation pipelines
* Human approval workflows for high-value leads
* Better structured lead-scoring policies
* Production database instead of local ChromaDB
* Automated CI/CD
* More comprehensive unit and integration tests

---

# 23. Future Improvements

The next version could introduce:

1. Real CRM synchronization
2. Real calendar scheduling
3. Multi-agent sales workflows
4. Lead prioritization
5. Automated follow-up emails
6. Sales analytics
7. Conversation memory
8. Human-in-the-loop approval
9. Agent evaluation and monitoring
10. Production authentication and access control

---



# 24. Repository

GitHub repository:

https://github.com/ankit-2244/AI-Sales-Agent

Live API:

https://ai-sales-agent.fastapicloud.dev/

Swagger API documentation:

https://ai-sales-agent.fastapicloud.dev/docs

---

# 25. Conclusion

The AI Sales Agent demonstrates an end-to-end agentic AI application that combines:

* LLMs
* RAG
* Hybrid retrieval
* BM25
* Reranking
* LangGraph
* Tool execution
* Lead qualification
* CRM interaction
* Demo booking
* FastAPI
* Cloud deployment

The project is designed as an MVP that can be extended into a production sales automation platform with real CRM, calendar, authentication, monitoring, and evaluation integrations.
