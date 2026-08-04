# 🗂️ Paper Trail

**A permission-aware enterprise policy search assistant**, built to demonstrate the core architecture behind AI-native enterprise search products: connectors, chunking, embeddings, permission-scoped retrieval, and LLM-powered synthesis.

Built as a hands-on portfolio project by Ashutosh Nagaria to explore the technical foundations behind roles like Glean's Connectors PM, directly extending real-world experience with ACL-scoped Knowledge Graph systems (People Finder, ServiceNow).

**🔗 Live demo: [enterprise-policy-connector.streamlit.app](https://enterprise-policy-connector.streamlit.app/)**
Try it as different personas (dropdown in the app) and watch permission enforcement happen in real time: same question, different access, different answer.

---

## The one-paragraph pitch

Most "AI search" demos are a thin LLM wrapper over a folder of PDFs, ask a question, get an answer, no notion of who's asking or what they're allowed to see. Paper Trail is built to demonstrate the actual infrastructure pattern behind real enterprise search products, where permission is checked before retrieval happens, not layered on as an afterthought. It connects to a real Google Drive folder, indexes real documents, and enforces zone- and role-based access control at query time, the same governance pattern products like Glean, Guru, and internal enterprise search tools are built around.

---

## The scenario

A fictional company, **DunderMifflin Enterprises** (yes, that paper company, used here purely as a fun, recognizable fictional backdrop with entirely original content, no copyrighted material), has HR policy documents, Travel, Hiring & Compensation, and Parental Leave, spread across Google Drive, organized by region: **Americas, APAC, and Europe**.

Paper Trail connects to that Drive folder, reads and indexes the content, and lets employees ask plain-English questions about policy, but it only ever answers using documents that specific employee is actually allowed to see.

**Try asking the same question as different personas** and watch both the answer and the access log change:
- **Pam Beesly** (Americas, Associate), only sees Americas-region policy
- **Oscar Martinez** (APAC, Senior), only sees APAC-region policy
- **Robert California** (Global, Executive/CFO), sees policy across all three regions

Try: *"What is the hotel budget for senior employees?"*, as Pam, then as Oscar. Same question, two completely different (and correctly scoped) answers.

---

## How it works, end to end

```
Google Drive  ->  Connector  ->  Chunking  ->  Embeddings  ->  Vector Index  ->  Permission Filter  ->  LLM Synthesis
 (source)         (pull &        (split into    (meaning        (Chroma)         (zone + role           (Claude)
                  normalize)      pieces)         fingerprint)                    check, pre-retrieval)
```

Here's what actually happens at each stage, explained in plain terms alongside the technical detail. The goal of this project was to understand and build every layer, not just call an API.

### 1. Connector: getting data out of the source system
A connector is the piece of software that authenticates against a source system (here, Google Drive), walks its folder structure, and pulls document content into a common format. This is conceptually the same job every enterprise search connector does, Slack, Confluence, Salesforce, whatever the source is, normalize whatever's out there into something the rest of the pipeline can process. Paper Trail's connector uses OAuth 2.0 to authenticate, lists the zone subfolders (Americas/APAC/Europe), and reads every document inside each one, including a `SCOPE` metadata line embedded at the top of each doc (`Zone=Americas | Sensitivity=General | Min-Role=Associate`) that the rest of the pipeline uses for permission enforcement.

### 2. Chunking: cutting documents into focused pieces
A single policy document might cover ten different topics across two pages. If you tried to search or embed the whole document as one block, you'd get a blurry, unhelpful "average meaning" that doesn't represent any single topic well. Chunking splits each document into smaller, ~500-character pieces (with slight overlap between them so a sentence never gets awkwardly cut in half), so each piece represents one focused idea: a single entitlement, a single exception rule, a single contact.

### 3. Embeddings: converting meaning into numbers
Each chunk is run through a small AI model that converts its meaning into a list of 384 numbers, sometimes called a "meaning fingerprint." Two chunks about similar topics end up with similar numbers, even if they use completely different words. This is what enables semantic search, "what's the travel budget" and "how much can I spend on flights" can match the same content, because the model understands meaning, not just keyword overlap. This runs entirely locally (`sentence-transformers`, `all-MiniLM-L6-v2`), no API call, no per-use cost, and no external dependency for what is by far the most frequently-run step in the pipeline.

### 4. Vector Index: a searchable home for every chunk
All those chunks, their embeddings, and their permission metadata get stored in ChromaDB, a database purpose-built for one job: given a new question's embedding, instantly find the stored chunks whose embeddings are closest to it. Think of it as the filing cabinet that makes the "meaning fingerprints" actually searchable at speed, rather than comparing a new question against every document from scratch every time.

### 5. Permission Filter: the part most demos skip
This is the core idea of the whole project. When a question comes in, the app knows which persona is asking (their zone and role). Before any retrieved chunk is allowed to reach the LLM or the user, it's checked against that persona's access: does the chunk's zone match (or is the persona "Global")? Does the persona's role meet the chunk's minimum role requirement? Chunks that fail either check are dropped, visibly, with a "N results blocked by permission check" indicator in the UI, so the enforcement is demonstrable, not hidden. Permission is enforced before retrieval, not by redacting an answer after the fact, the same pattern real enterprise search governance requires, and a common gap in naive RAG implementations.

### 6. Synthesis: where the LLM actually comes in
Only the permission-approved chunks get sent to Claude (Haiku), along with the original question and an instruction to answer using only the provided excerpts and cite the source document. This is the only step in the entire pipeline that uses a paid, hosted LLM, everything upstream (chunking, embedding, indexing, permission filtering) is deterministic code or a small local model. That's a deliberate architectural choice: reserve the expensive, high-reasoning model for the one step that actually needs reasoning, and keep everything else cheap and fast.

### 7. Audit Log: governance, not just retrieval
Every query is logged: who asked, what they asked, what they were shown, and how many results were blocked. This mirrors a real requirement in enterprise search, compliance and security teams need to be able to answer "who accessed what, and when," not just "did the search work."

---

## Why these design choices (the PM judgment behind the build)

**Local embeddings over a paid API for the highest-volume step.** Embeddings run on every chunk, every time the index is built, using a free, local model keeps that step at zero marginal cost. The paid LLM is reserved only for final answer synthesis, the one place genuine reasoning is required. This mirrors how production systems are actually cost-optimized: you don't run an expensive model for a job a cheap one can do.

**Permission encoded as document metadata, not folder location.** An earlier version of this project considered a folder-based permission model (a `General/` folder vs. an `HR-Only/` folder). That was deliberately abandoned in favor of a `SCOPE` tag read from each document's content, checked live at query time, because real enterprise content doesn't stay neatly organized by permission level; it gets moved, shared, and reorganized. Permission-as-metadata, checked at retrieval time, is closer to how production access control actually has to work.

**Retrieval-time filtering, not display-time redaction.** A common and risky anti-pattern in RAG systems is to retrieve everything and hide what a user shouldn't see in the UI, meaning the sensitive content still entered the LLM's context. Paper Trail filters chunks out before they're ever considered for retrieval or shown to the LLM, so blocked content never enters the answer-generation pipeline at all.

---

## Project structure

```
paper-trail/
├── connector/
│   └── drive_connector.py       # OAuth + Drive API walk + content extraction
├── indexing/
│   ├── build_index.py            # Chunk -> embed -> store pipeline
│   ├── chunk_and_embed.py        # Standalone chunking/embedding test script
│   ├── peek_index.py             # Inspect what's stored in the vector DB
│   └── search_with_persona.py    # CLI-based permission-aware search test
├── app/
│   ├── query_app.py               # Streamlit UI, the live app
│   └── assets/                    # Background image, static assets
├── data/
│   ├── chroma_store/              # Persisted vector index (checked in for hosted deploy)
│   └── query_log.txt              # Audit log of all queries
├── .streamlit/
│   └── config.toml                # Theme config
├── requirements.txt
├── runtime.txt                    # Pins Python version for deployment
└── .gitignore                     # Excludes secrets, credentials, venv
```

---

## Running it locally

**1. Clone and set up environment**
```bash
git clone https://github.com/YOUR_USERNAME/paper-trail.git
cd paper-trail
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Google Drive connector setup**
- Create a Google Cloud project, enable the Drive API
- Configure an OAuth consent screen (External, add yourself as a test user)
- Create OAuth credentials (Desktop app type), download the JSON
- Save as `credentials/client_secret.json`

**3. Environment variables**
Create a `.env` file in the root:
```
ANTHROPIC_API_KEY=your_key_here
```

**4. Build the index**
```bash
python indexing/build_index.py
```

**5. Run the app**
```bash
streamlit run app/query_app.py
```

---

## What I'd build next

- **Role-level chunk tagging**: permission is currently enforced at the zone level; any document is readable in full by any role at or above its minimum. A natural next step is per-section role tagging (an Associate persona can retrieve the general parts of a policy but never the Executive compensation section) for finer-grained ACL enforcement, right now that's designed into the metadata schema but not yet demonstrated with a genuinely role-restricted document.
- **Incremental sync**: the current indexing job does a full refresh of all documents; a production version would only re-process documents that actually changed since the last run.
- **Graph-based retrieval**: a companion project (in progress) explores Neo4j and Cypher for relationship-based queries ("who works with whom," "what's connected to what") that vector similarity search fundamentally can't answer, the other half of how systems like Glean's Enterprise Graph combine retrieval with relationship traversal.

---

## Background

This project translates PM experience with ACL-scoped Knowledge Graph products (People Finder, ServiceNow) into hands-on, working infrastructure: a real connector, real embeddings, and real permission-aware retrieval, the technical core of AI-native enterprise search platforms.

Built with Claude, Google Drive API, ChromaDB, Sentence Transformers, and Streamlit.
