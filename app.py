import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def extract_pdf_chunks(pdf_file, max_len=500):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        for para in paragraphs:
            chunks.append((page_num, para[:max_len]))
    return chunks

st.set_page_config(page_title="📘 PDF Chatbot", layout="centered")
st.markdown("## 🤖 بوت ذكاء يقرأ PDF ويجيب على أسئلتك")

uploaded_file = st.file_uploader("📎 ارفع ملف PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 جارٍ قراءة الملف..."):
        pdf_chunks = extract_pdf_chunks(uploaded_file)
        paragraphs_text = [chunk[1] for chunk in pdf_chunks]
        embeddings = model.encode(paragraphs_text, convert_to_tensor=True)

    question = st.text_input("📝 اسأل سؤالاً عن محتوى PDF:")

    if question:
        with st.spinner("🤔 يُفكّر..."):
            question_embedding = model.encode(question, convert_to_tensor=True)
            hits = util.semantic_search(question_embedding, embeddings, top_k=1)[0]
            hit = hits[0]
            idx = hit["corpus_id"]
            score = hit["score"]
            page, para = pdf_chunks[idx]
            st.markdown(f"📄 من الصفحة {page} (درجة التطابق: {score:.2f})")
            st.info(para)
