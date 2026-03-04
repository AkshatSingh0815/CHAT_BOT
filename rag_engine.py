import os
import time
from collections import OrderedDict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv()

class RAGEngine:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

        # ---------------------------
        # Build Vectorstore
        # ---------------------------
        self.vectorstore = self.build_vectorstore()

        # ---------------------------
        # LLM (Groq)
        # ---------------------------
        self.llm = ChatGroq(
            api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        # ---------------------------
        # Retriever (MMR)
        # ---------------------------
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        # ---------------------------
        # Custom PM-JAY Prompt (Chiti)
        # ---------------------------
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are Chiti, an intelligent assistant helping users with claim-related queries.

Your name is Chiti and you are a helpful assistant for PM-JAY claim-related questions developed by Akshat Singh.

Rules:
1. Use ONLY the information provided in the CONTEXT section.
2. Do NOT provide any information that is not present in the context.
3. If the answer cannot be found in the context, reply:
"I’m sorry, I could not find relevant information in the available records."
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

CONTEXT:
{context}

User Question:
{question}

Answer:
"""
        )

        # ---------------------------
        # RetrievalQA Chain
        # ---------------------------
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": self.prompt_template
            }
        )

        # ---------------------------
        # LRU + TTL Cache
        # ---------------------------
        self.cache = OrderedDict()
        self.cache_limit = 100
        self.cache_ttl = 60 * 60 * 3  # 3 hours

        # ---------------------------
        # Analytics
        # ---------------------------
        self.total_queries = 0
        self.cache_hits = 0

    # ============================================================
    # Build Vectorstore
    # ============================================================

    def build_vectorstore(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    # ============================================================
    # Cache Maintenance
    # ============================================================

    def clean_expired_cache(self):
        current_time = time.time()
        expired_keys = []

        for key, value in self.cache.items():
            if current_time - value["timestamp"] > self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

    def enforce_lru_limit(self):
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    # ============================================================
    # Answer Formatter (Extra Safety Layer)
    # ============================================================

    def format_answer(self, text):
        text = text.replace("•", "")
        text = text.replace("* ", "")
        text = text.replace("- ", "")

        paragraphs = text.split("\n")
        clean_paragraphs = []

        for p in paragraphs:
            p = p.strip()
            if p:
                clean_paragraphs.append(p)

        return "\n\n".join(clean_paragraphs)

    # ============================================================
    # Main Answer Function
    # ============================================================

    def answer(self, query):
        self.total_queries += 1

        # Clean expired cache
        self.clean_expired_cache()

        # Cache Check (Exact Match)
        if query in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(query)
            return self.cache[query]["response"]

        # Generate Response via RAG
        response = self.qa_chain.invoke({"query": query})
        answer = response["result"]

        formatted_answer = self.format_answer(answer)

        # Store in Cache
        self.cache[query] = {
            "response": formatted_answer,
            "timestamp": time.time()
        }

        self.enforce_lru_limit()

        return formatted_answer

    # ============================================================
    # Analytics
    # ============================================================

    def get_analytics(self):
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (
                round((self.cache_hits / self.total_queries) * 100, 2)
                if self.total_queries > 0 else 0
            ),
            "current_cache_size": len(self.cache)
        }