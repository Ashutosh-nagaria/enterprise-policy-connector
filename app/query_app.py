import sys
import os
os.environ['STREAMLIT_WATCHER_TYPE'] = 'poll'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import base64
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()  # this reads your .env file and makes ANTHROPIC_API_KEY available

# --- Personas, same as Chapter 5 ---
ROLE_LEVELS = {'Associate': 1, 'Senior': 2, 'Executive': 3}
PERSONAS = {
    'Pam Beesly (Americas, Associate)': {'name': 'Pam Beesly', 'zone': 'Americas', 'role': 'Associate'},
    'Jan Levinson (Americas, Senior)': {'name': 'Jan Levinson', 'zone': 'Americas', 'role': 'Senior'},
    'Oscar Martinez (APAC, Senior)': {'name': 'Oscar Martinez', 'zone': 'APAC', 'role': 'Senior'},
    'Robert California (Global, CFO/Executive)': {'name': 'Robert California', 'zone': 'Global', 'role': 'Executive'},
}

# --- Chapter 7: suggested queries shown as clickable pills ---
SUGGESTED_QUERIES = [
    {"label": "Travel policy (APAC)", "full_query": "What's the travel policy for Associates in APAC?"},
    {"label": "Parental leave (Europe)", "full_query": "What's the parental leave policy in Europe?"},
    {"label": "CFO's travel budget", "full_query": "What's the CFO's travel budget?"},  # role-gated example
    {"label": "Remote work policy", "full_query": "What's the remote work policy?"},    # out-of-scope example
]

@st.cache_resource  # load these once, not on every single question, keeps the app fast
def load_model_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="data/chroma_store")
    collection = client.get_or_create_collection(name="policybot_docs")
    return model, collection

def is_allowed(persona, chunk_metadata):
    zone_ok = (persona['zone'] == 'Global') or (persona['zone'] == chunk_metadata['zone'])
    role_ok = ROLE_LEVELS.get(persona['role'], 0) >= ROLE_LEVELS.get(chunk_metadata['min_role'], 1)
    return zone_ok and role_ok

def is_small_talk(question):
    """
    Quick, free, deterministic check for greetings and chatter that
    shouldn't trigger a policy search at all. Runs before any retrieval.
    """
    small_talk_phrases = {
        'hi', 'hey', 'hello', 'hiya', 'yo',
        'thanks', 'thank you', 'ok', 'okay', 'cool', 'got it',
        'how are you', "how's it going", 'good morning', 'good afternoon',
        'good evening', 'bye', 'goodbye', 'see you',
    }
    cleaned = question.strip().lower().rstrip('!?.')
    return cleaned in small_talk_phrases or len(cleaned) <= 3

def log_query(persona_name, question, allowed_docs, blocked_count):
    """Writes one line per query to an audit log file — the 'security camera' log we talked about."""
    os.makedirs('data', exist_ok=True)
    with open('data/query_log.txt', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {persona_name} asked: \"{question}\" | Shown: {allowed_docs} | Blocked: {blocked_count} chunks\n")

def ask_claude(question, context_chunks):
    """Sends the question + the permission-filtered chunks to Claude, asks it to answer using only that context."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    client = anthropic.Anthropic(api_key=api_key)

    context_text = "\n\n---\n\n".join([f"[Source: {c['doc_name']}]\n{c['text']}" for c in context_chunks])

    prompt = f"""You are Paper Trail, an internal policy assistant for DunderMifflin Enterprises.
Answer the employee's question using ONLY the policy excerpts provided below.
If the excerpts don't contain the answer, say so clearly rather than guessing.
Always mention which document(s) your answer is based on.

POLICY EXCERPTS:
{context_text}

EMPLOYEE QUESTION: {question}

ANSWER:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# --- Background image setup ---
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_path = os.path.join(os.path.dirname(__file__), "assets", "background.jpg")
bg_image = get_base64_image(bg_path)

# --- The actual page ---
st.set_page_config(page_title="Paper Trail", page_icon="🗂️", layout="wide")

st.markdown(f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background-color: transparent;
    }}
    div[data-testid="stDecoration"] {{
        display: none;
    }}
    h1, h2, h3 {{
        font-family: 'Georgia', serif !important;
        color: #4a3f35 !important;
    }}
    div[data-testid="stMarkdownContainer"] p, .stCaption, label {{
        font-family: 'Georgia', serif !important;
        color: #4a3f35;
    }}
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        border: 1px solid #a99a85 !important;
        border-radius: 6px !important;
        font-family: 'Georgia', serif !important;
    }}
    .stTextInput, .stSelectbox {{
        max-width: 500px;
    }}
    .stButton button {{
        background-color: #4a3f35 !important;
        color: #f5eee3 !important;
        border-radius: 6px !important;
        border: none !important;
        font-family: 'Georgia', serif !important;
        font-weight: bold;
        padding: 0.5em 1.5em !important;
    }}
    .stButton button p {{
        color: #f5eee3 !important;
    }}
    .stButton button:hover {{
        background-color: #a99a85 !important;
        color: #4a3f35 !important;
    }}
    .stButton button:hover p {{
        color: #4a3f35 !important;
    }}
    div[data-testid="stExpander"] {{
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid #d8ccb8;
        border-radius: 8px;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 8px;
        font-family: 'Georgia', serif !important;
    }}
    .persona-context {{
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid #d8ccb8;
        border-radius: 8px;
        padding: 0.6em 1em;
        margin: 0.5em 0 1.2em 0;
        font-family: 'Georgia', serif;
        color: #4a3f35;
        max-width: 500px;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🗂️ Enterprise Policy Connector")
st.caption("Permission-aware policy search, built for DunderMifflin Enterprises")

persona_label = st.selectbox("I am:", list(PERSONAS.keys()))
persona = PERSONAS[persona_label]

# --- Chapter 7: visible persona/zone/role context, so the ACL logic isn't hidden ---
st.markdown(f"""
<div class="persona-context">
🪪 Viewing as <strong>{persona['name']}</strong> &nbsp;·&nbsp; Zone: <strong>{persona['zone']}</strong> &nbsp;·&nbsp; Role: <strong>{persona['role']}</strong>
</div>
""", unsafe_allow_html=True)

# --- Chapter 7: session state setup for pill clicks ---
if 'question_input' not in st.session_state:
    st.session_state.question_input = ''
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False

st.caption("Try asking:")
pill_cols = st.columns(len(SUGGESTED_QUERIES))
for col, suggestion in zip(pill_cols, SUGGESTED_QUERIES):
    with col:
        if st.button(suggestion["label"], key=f"pill_{suggestion['label']}", use_container_width=True):
            st.session_state.question_input = suggestion["full_query"]
            st.session_state.auto_run = True
            st.rerun()

question = st.text_input("Ask a policy question:", key='question_input')

ask_clicked = st.button("Ask")
should_run = (ask_clicked and question) or st.session_state.auto_run

if should_run and question:
    st.session_state.auto_run = False  # reset so it doesn't re-fire on the next unrelated rerun

    if is_small_talk(question):
        st.markdown("### Answer")
        st.write("Hi there! I'm Paper Trail, DunderMifflin's policy assistant. Ask me about travel, hiring, or parental leave policies and I'll do my best to help.")
    else:
        model, collection = load_model_and_db()

        with st.spinner("Searching policies..."):
            query_embedding = model.encode([question])[0]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=10,
                include=['documents', 'metadatas'],
            )

            allowed_chunks = []
            blocked_count = 0
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                text = results['documents'][0][i]
                if is_allowed(persona, meta):
                    allowed_chunks.append({'doc_name': meta['doc_name'], 'text': text})
                else:
                    blocked_count += 1

        if not allowed_chunks:
            st.warning("No policy content found that you have access to for this question.")
        else:
            with st.spinner("Generating answer..."):
                top_chunks = allowed_chunks[:5]  # don't overload the LLM with too much context
                answer = ask_claude(question, top_chunks)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("📄 Sources used"):
                seen = set()
                for c in top_chunks:
                    if c['doc_name'] not in seen:
                        st.write(f"- {c['doc_name']}")
                        seen.add(c['doc_name'])

            if blocked_count > 0:
                st.caption(f"🔒 {blocked_count} additional result(s) existed but were outside your zone/role access.")

            log_query(persona['name'], question, list(seen), blocked_count)

            st.divider()

with st.expander("Admin: View Query Log"):
    entered_code = st.text_input("Enter admin passcode", type="password")
    if entered_code:
        if entered_code == os.getenv('ADMIN_PASSCODE'):
            log_path = 'data/query_log.txt'
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_contents = f.read()
                if log_contents.strip():
                    st.text_area("Query log", log_contents, height=300)
                else:
                    st.info("No queries logged yet.")
            else:
                st.info("No queries logged yet.")
        else:
            st.error("Incorrect passcode.")