import os
import time
from collections import OrderedDict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA           #  fixed: langchain not langchain_classic
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document      #  fixed: langchain_core not langchain_classic
from dotenv import load_dotenv
from pydantic import SecretStr
from ocr_pipeline import extract_all_image_text
import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
load_dotenv()
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OCR_CACHE = os.path.join(BASE_DIR, "storage", "ocr_cache", "ocr_text.txt")
VECTOR_DB = os.path.join(BASE_DIR, "storage", "vector_db")
class RAGEngine:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vectorstore = self.build_vectorstore()
        self.llm = ChatGroq(
            api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are Chiti, a PM-JAY claims assistant.
Use only the provided context.
If answer not found say:
"I'm sorry, I could not find relevant information."
Rules:
1.Answer me in HTML format with appropriate tags.
1. Use ONLY the information provided in the CONTEXT section.
2. Do NOT provide any information that is not present in the context.
3. If the answer cannot be found in the context, reply:
   "I'm sorry, I could not find relevant information in the available records."
4. Do NOT copy text verbatim from the context; always rephrase clearly and professionally.
5. Remove duplicates and redundant information.
6. Keep the response concise, clear, and well-structured.
7. Understand minor spelling mistakes and grammatical errors.
8. Detect the language of the user's question and respond ONLY in the SAME language.
Formatting Rules:
- Do NOT use bullet points.
- Do NOT use markdown formatting.
- Write short paragraphs.
- Each paragraph must be separated by a blank line.
- Each paragraph must contain 2-4 sentences only.
use the exact formal below:
<ol>
    <ul>
        <h3>heading</h3>
        <li>point 1</li>
        <li>point 2</li>
    </ul> 
</ol>
continue the order of the points as per the context and do not change the order of the points.

Context:
{context}
Question:
{question}
Answer:
"""
#--------------------------------------------------------------------------------------------------------------------------------
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        self.cache: OrderedDict = OrderedDict()
        self.cache_limit = 100
        self.cache_ttl   = 60 * 60 * 3   # 3 hours
        self.total_queries = 0
        self.cache_hits    = 0
    # ----------------------------------------------------------------------------------------------------------------------------
    def build_vectorstore(self):
        print("Building vector store...")
        os.makedirs(os.path.dirname(OCR_CACHE), exist_ok=True)
        os.makedirs(VECTOR_DB, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        if os.path.exists(os.path.join(VECTOR_DB, "index.faiss")):
            print("Loading existing vector store from disk...")
            return FAISS.load_local(
                VECTOR_DB, embeddings, allow_dangerous_deserialization=True
            )
        #----------------------------------------------------------------------------------------------------------------------------
        # --- PDF text chunks ---
        print("Loading and splitting PDF...")
        loader   = PyPDFLoader(self.pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = loader.load_and_split(text_splitter=splitter)
        print(f"PDF chunks: {len(chunks)}")
        # --- OCR text ---
        if os.path.exists(OCR_CACHE):
            with open(OCR_CACHE, "r", encoding="utf-8") as f:
                ocr_text = f.read()
            print("OCR text loaded from cache.")
        else:
            print("Running OCR pipeline (this may take a while)...")
            ocr_text = extract_all_image_text(self.pdf_path)
            with open(OCR_CACHE, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            print("OCR text extracted and cached successfully.")

        if ocr_text.strip():
            print("Splitting OCR text...")
            ocr_doc    = Document(page_content=ocr_text, metadata={"source": "ocr"})
            ocr_chunks = splitter.split_documents([ocr_doc])
            chunks.extend(ocr_chunks)
            print(f"OCR chunks added: {len(ocr_chunks)}")

        print(f"Total chunks: {len(chunks)}")

        print("Generating embeddings and saving vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB)
        print("Vector store built and saved successfully.")
        return vectorstore

    # ------------------------------------------------------------------
    def _is_cache_valid(self, entry: dict) -> bool:
        """Return True if the cache entry has not expired."""
        return (time.time() - entry["timestamp"]) < self.cache_ttl

    # ------------------------------------------------------------------
    def answer(self, query: str) -> str:
        self.total_queries += 1

        # Check cache (with TTL enforcement)
        if query in self.cache:
            entry = self.cache[query]
            if self._is_cache_valid(entry):
                self.cache_hits += 1
                # Move to end to keep it 'recently used'
                self.cache.move_to_end(query)
                return entry["response"]
            else:
                # Expired — remove it
                del self.cache[query]

        # Query the LLM
        response = self.qa_chain.invoke({"query": query})
        answer   = response["result"]

        # Evict oldest entry if over limit
        if len(self.cache) >= self.cache_limit:
            self.cache.popitem(last=False)

        self.cache[query] = {
            "response":  answer,
            "timestamp": time.time()
        }

        return answer

    # ------------------------------------------------------------------
    def get_analytics(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "cache_hits":    self.cache_hits,
            "cache_hit_rate": (
                f"{(self.cache_hits / self.total_queries * 100):.1f}%"
                if self.total_queries > 0 else "N/A"
            )
        }