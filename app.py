import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Viva Practice Room..",page_icon="🎓",layout="wide")
st.title(" AI Viva Practice Room 🎓")

@st.cache_resource
def load_tools():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="viva_db")
    collection = client.get_or_create_collection(
        "notes")
    return embedding_model, collection
embedding_model, collection = load_tools()

def chunk_text(text, size=500):
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > size and current:
            chunks.append(current)
            current = para
        else:
            current = (
                current + "\n" + para
            ).strip()
    if current:
        chunks.append(current)
    return chunks

def save_notes(text):
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embedding_model.encode(
        chunks
    ).tolist()
    existing_count = collection.count()
    ids = [
        f"note_{existing_count + i}"
        for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    return len(chunks)

def ask_ollama(prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]

def new_question(difficulty, mode):
    data = collection.get()
    if not data["documents"]:
        return None
    context = "\n---\n".join(
        data["documents"][:5]
    )
    prompt = f"""
You are an experienced viva examiner.
Create ONE viva question using ONLY the study material below.
Difficulty: {difficulty}
Viva Mode: {mode}

Rules:
- Ask only ONE question.
- Do not give the answer.
- Do not explain the question.
- Make the question suitable for a college viva.
- Avoid repeating obvious questions.

STUDY MATERIAL:
{context}
Return only the question.
"""
    return ask_ollama(prompt)

def evaluate(question, answer):
    query = embedding_model.encode(
        [question + " " + answer]
    ).tolist()
    hits = collection.query(
        query_embeddings=query,
        n_results=min(3, collection.count())
    )
    context = "\n---\n".join(
        hits["documents"][0]
    )
    prompt = f"""
You are a strict but friendly viva examiner.
Evaluate the student's answer using the study material
as the source of truth.

STUDY MATERIAL:
{context}

QUESTION:
{question}

STUDENT ANSWER:
{answer}

Reply exactly in this format:
Score: X/10

What was good:
- Point 1
- Point 2

What was missing:
- Point 1
- Point 2

Correction:
Give a short correction if necessary.

Ideal answer:
Give a clear 3-4 line answer.

Examiner comment:
Give a short,motivating and encouraging comment.
"""

    return ask_ollama(prompt)

def get_hint(question):
    data = collection.get()
    if not data["documents"]:
        return None
    context = "\n---\n".join(
        data["documents"][:3]
    )
    prompt = f"""
You are helping a student during viva preparation.

Question:
{question}

Study material:
{context}

Give ONE short hint.

Do not give the complete answer.
"""

    return ask_ollama(prompt)

if "question" not in st.session_state:
    st.session_state["question"] = None

if "feedback" not in st.session_state:
    st.session_state["feedback"] = None

if "hint" not in st.session_state:
    st.session_state["hint"] = None

if "score" not in st.session_state:
    st.session_state["score"] = []

if "started" not in st.session_state:
    st.session_state["started"] = False
st.write(
    "The best way to prepare for your viva is by uploading your study material, practicing viva questions, "
    "and receiving AI-powered feedback on your answers."
)
st.divider()
st.header(" Add Your Study Material 📚")
tab1, tab2 = st.tabs(
    ["Upload File📄 ", " Paste Notes📝"]
)
with tab1:
    uploaded = st.file_uploader(
        "Upload your study material",
        type=["txt"],
        help="Upload a .txt file containing your notes."
    )
    if uploaded:
        file_text = uploaded.read().decode(
            "utf-8"
        )
        st.success(
            f"File loaded: {uploaded.name}"
        )
        st.text_area(
            "Preview",
            file_text,
            height=180,
            disabled=True
        )
with tab2:
    notes = st.text_area(
        "Paste your study notes here",
        height=220,
        placeholder="Paste your subject notes here..."
    )
if st.button(
    " Save & Process Study Material💾",
    type="primary"
):
    if uploaded:
        text = file_text
    else:
        text = notes
    if text and text.strip():
        with st.spinner("Processing your study material..."):
            n = save_notes(text)
        st.success(
            f"✅ Study material processed successfully! "
            f"{n} chunks created🙌."
        )
        st.session_state["started"] = True
    else:
        st.warning(
            "Please upload a file or paste your notes first."
        )
if collection.count() > 0:
    st.info(
        f" {collection.count()} study chunks📚"
        f"are currently available for your viva."
    )

st.divider()
st.header("Viva Settings⚙️")
col1, col2, col3 = st.columns(3)
with col1:
    difficulty = st.selectbox(
        "🎯 Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )
with col2:

    mode = st.selectbox(
        "Viva Mode🧑‍🏫 ",
        [
            "Normal Viva",
            "Rapid Fire",
            "Technical Interview"
        ]
    )
with col3:

    total_questions = st.selectbox(
        "Questions limit🫟",
        [5, 10, 15, 20]
    )

st.divider()
st.header("Start Your Viva🎤")
if collection.count() == 0:
    st.warning(
        "Please add and process your study material first⚠️ ."
    )
else:
    if st.button(
        " Start Viva Practice🚀",
        type="primary"
    ):
        with st.spinner(
            "please wait while Examiner is thinking..."
        ):
            q = new_question(
                difficulty,
                mode
            )
        st.session_state["question"] = q
        st.session_state["feedback"] = None
        st.session_state["hint"] = None

if st.session_state["question"]:
    st.divider()
    st.header(" Viva Question")
    st.info(
        st.session_state["question"]
    )
    if st.button(" want a Hint..💡"):
        with st.spinner(
            "wait a moment while preparing hint..."
        ):
            hint = get_hint(
                st.session_state["question"]
            )
        st.session_state["hint"] = hint
    if st.session_state["hint"]:
        st.warning(
            "💡 Hint: "
            + st.session_state["hint"]
        )
    answer = st.text_area(
        "🗣️ Your Answer",
        height=180,
        placeholder="Type your answer here..."
    )
    if st.button(
        "Submit Answer✅ ",
        type="primary"
    ):
        if answer.strip():
            with st.spinner(
                "wait a moment while Examiner is evaluating..."
            ):
                feedback = evaluate(
                    st.session_state["question"],
                    answer
                )
            st.session_state["feedback"] = feedback
        else:
            st.warning(
                "Please type your answer here..."
            )

if st.session_state["feedback"]:
    st.divider()
    st.header(" Viva Feedback 📊")
    st.write(
        st.session_state["feedback"]
    )
    if st.button(
        "➡️ Next Question"
    ):
        with st.spinner(
            "Preparing next question..."
        ):
            q = new_question(
                difficulty,
                mode
            )
        st.session_state["question"] = q
        st.session_state["feedback"] = None
        st.session_state["hint"] = None
        st.rerun()

with st.sidebar:
    st.header("Viva Dashboard 📊")
    st.metric(
        "Study Chunks",
        collection.count()
    )
    st.metric(
        "Questions Answered",
        len(st.session_state["score"])
    )
    st.divider()
    st.subheader("📌 Features")
    st.write("✅ AI-generated viva questions")
    st.write("✅ Difficulty levels")
    st.write("✅ Multiple viva modes")
    st.write("✅ AI answer evaluation")
    st.write("✅ Hints")
    st.write("✅ Study-material-based answers")
    st.write("✅ Next question")
    st.write("✅ Performance tracking")
    st.divider()
    if st.button(
        "Clear All Study Material 🗑️ "
    ):
        data = collection.get()
        if data["ids"]:
            collection.delete(
                ids=data["ids"]
            )
        st.session_state.clear()
        st.rerun()
if st.button("EXIT🚪"):
    st.session_state["question"] = None
    st.session_state["feedback"] = None
    st.session_state["hint"] = None
    st.session_state["started"] = False
    st.success("successfully completed the Viva Practice Room 👋")