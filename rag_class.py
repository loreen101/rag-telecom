import os
import re
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from streamlit import text


DATA_PATH = r"..\data"
CACHE_DIR = r"..\cache"
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")
CHUNKS_CACHE_PATH = os.path.join(CACHE_DIR, "chunks_metadata.pkl")
EMBEDDING_MODEL = "omarelshehy/arabic-english-sts-matryoshka"
LLM_MODEL = "llama-3.1-8b-instant"
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


class TelecomRAG:
    def __init__(self):
        print("Initializing TelecomRAG system...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.metadata = []
        self.index = None
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # Try to load from cache first
        if self.load_from_cache():
            print("✓ Loaded from cache! Skipping data processing.")
        else:
            print("⚠ Cache not found or invalid. Processing data...")
            self.load_data()
            self.save_to_cache()

    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================
    def load_from_cache(self):
        """Load chunks, metadata, and FAISS index from cache."""
        try:
            # Check if cache files exist
            if not os.path.exists(CHUNKS_CACHE_PATH) or not os.path.exists(FAISS_INDEX_PATH):
                return False
            
            # Load chunks and metadata
            with open(CHUNKS_CACHE_PATH, 'rb') as f:
                cache_data = pickle.load(f)
                self.chunks = cache_data['chunks']
                self.metadata = cache_data['metadata']
            
            # Load FAISS index
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            
            print(f"✓ Cache loaded: {len(self.chunks)} chunks, {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            print(f"Cache load failed: {e}")
            return False
    
    def save_to_cache(self):
        """Save chunks, metadata, and FAISS index to cache."""
        try:
            # Save chunks and metadata
            with open(CHUNKS_CACHE_PATH, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.metadata
                }, f)
            
            # Save FAISS index
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            
            print(f"✓ Cache saved: {len(self.chunks)} chunks, {self.index.ntotal} vectors")
        
        except Exception as e:
            print(f"Cache save failed: {e}")

    # ============================================================
    # LOAD & PROCESS DATA
    # ============================================================
    def load_data(self):
        print("Loading and processing documents...")

        for file in os.listdir(DATA_PATH):
            if file.endswith(".md"):
                with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
                    text = f.read()

                doc_chunks = self.chunk_text(text)

                for chunk in doc_chunks:
                    self.chunks.append(chunk)
                    self.metadata.append({"source": file})

        embeddings = self.create_embeddings(self.chunks)
        self.index = self.build_faiss_index(embeddings)

    # ============================================================
    # CHUNKING
    # ============================================================
    def chunk_text(self, text):
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks, current_chunk = [], ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > 700:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # ============================================================
    # EMBEDDINGS
    # ============================================================
    def create_embeddings(self, chunks):
        """Converts text chunks into vector embeddings."""
        print("Creating embeddings...")

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        print("Embeddings created successfully!")
        print(f"Embedding shape: {embeddings.shape}")

        return embeddings

    # ============================================================
    # FAISS INDEX
    # ============================================================
    def build_faiss_index(self, embeddings):
        """Builds FAISS index for fast similarity search."""
        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        print("FAISS index built successfully!")
        print(f"Total vectors: {faiss_index.ntotal}")

        return faiss_index

    # ================== RETRIEVE ==================
    def retrieve(self, query, top_k=6):
        print(f"Searching for: {query}")

        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score > 0.4:
                results.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(score)
                })

        print(f"Found {len(results)} chunks")
        return results

    def normalize_arabic_text(self, text):
        """Normalizes Arabic spelling and punctuation for keyword routing."""
        normalized = text.strip().lower()
        normalized = re.sub(r"[إأآا]", "ا", normalized)
        normalized = re.sub(r"[ىي]", "ي", normalized)
        normalized = re.sub(r"ة", "ه", normalized)
        normalized = re.sub(r"[ؤئ]", "ء", normalized)
        normalized = re.sub(r"[ًٌٍَُِّْـ]", "", normalized)
        normalized = normalized.replace("؟", "?")
        normalized = re.sub(r"[^\w\s?]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def contains_any_phrase(self, text, phrases):
        return any(phrase in text for phrase in phrases)
    
    def route_query(self, query):
        """
        Decides how to handle the user query before sending it to the LLM.

        Returns one of four labels:
            "casual_conv"   -> simple greetings/small talk (no LLM needed)
            "inquiry"       -> normal telecom question (needs RAG + LLM)
            "out_of_scope"  -> unrelated to telecom support
            "ticket"        -> user is asking to create a ticket or escalate
        """
        q = self.normalize_arabic_text(query)

        ticket_phrases = [
            "اشتكي", 
            "اعمل تذكره", "افتح تذكره", "افتح شكوي", "سجل شكوي", "قدم شكوي",
            "ارفع شكوي", "رفع شكوي", "اعمل شكوى", "افتح بلاغ", "صعد", "تصعيد",
            "حول للدعم", "حول للمهندس", "ارسل مهندس", "ابعت مهندس", "ابعث مهندس",
            "عايز مهندس", "عاوز مهندس", "نزول مهندس", "متابعه مع الدعم",
            "ticket", "open ticket", "create ticket", "raise ticket",
            "escalate", "escalation", "complaint", "raise complaint",
            "dispatch engineer", "send engineer"
        ]

        telecom_phrases = [
            "نت", "انترنت", "الشبكه", "شبكه", "الراوتر", "راوتر", "واي فاي",
            "wifi", "5g", "4g", "فايبر", "fiber", "باقه", "الباقه", "خط",
            "الخط", "شريحه", "sim", "رصيد", "فاتوره", "فاتوره", "سرعه",
            "بطء", "بطيء", "مقطوع", "التطبيق", "الابلكيشن", "تغطيه", "روامينج",
            "تفعيل", "الغاء", "عرض", "العروض", "فاتوره", "billing", "refund",
            "invoice", "coverage", "network", "mobile", "data", "throttling",
            "outage", "ont", "roaming", "sla"
        ]

        casual_conv_phrases = [
            "ازيك", "اهلا", "مرحبا", "السلام عليكم", "هاي", "hello", "hi",
            "thanks", "thank you", "شكرا", "متشكر", "صباح الخير"
        ]

        out_of_scope_phrases = [
            "طقس", "درجات الحراره", "ماتش", "كوره", "طبخ", "وصفه", "فيلم",
            "اغنيه", "weather", "football", "movie", "recipe", "restaurant",
            "bitcoin"
        ]

        # Priority 1: Check for ticket requests
        if self.contains_any_phrase(q, ticket_phrases):
            return "ticket"

        # Priority 2: Check for casual conversation (simple greetings - don't use LLM)
        if self.contains_any_phrase(q, casual_conv_phrases):
            return "casual_conv"

        # Priority 3: Check for out-of-scope (but not telecom-related)
        if self.contains_any_phrase(q, out_of_scope_phrases) and not self.contains_any_phrase(q, telecom_phrases):
            return "out_of_scope"

        # Priority 4: Check for telecom inquiry
        if self.contains_any_phrase(q, telecom_phrases):
            return "inquiry"

        # Priority 5: Very short queries default to casual
        if len(q.split()) <= 3:
            return "casual_conv"

        return "out_of_scope"

    # ================== CLEAN REPETITION ==================
    def remove_repetition(self, text):
        lines = text.split("\n")
        seen = set()
        clean = []
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                clean.append(line)
                seen.add(line)
        return "\n".join(clean)

    # ================== GENERATE ==================
    def parse_llm_response(self, raw_text):
        """Extracts answer and needs_action even if formatting varies slightly."""
        cleaned = raw_text.strip()

        needs_action_match = re.search(
            r"needs[_\s-]*action\s*[:=]\s*(yes|no)",
            cleaned,
            flags=re.IGNORECASE
        )
        needs_action = needs_action_match.group(1).upper() if needs_action_match else "NO"

        answer_match = re.search(
            r"answer\s*[:=]\s*(.*?)(?=\n\s*needs[_\s-]*action\s*[:=]|\Z)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL
        )

        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = re.sub(
                r"\n?\s*needs[_\s-]*action\s*[:=]\s*(yes|no)\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE
            ).strip()

        if not answer:
            answer = "مش متأكد من البيانات المتاحة يا فندم."

        return answer, needs_action

    
    def generate_answer(self, query, retrieved_results):
        if not retrieved_results:
            return {
                "answer": "مش متأكد من البيانات المتاحة يا فندم.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "Unknown"
            }

        context = "\n\n".join([
            f"Source: {res['source']}\n{res['text']}"
            for res in retrieved_results
        ])
        system_prompt = """أنت مساعد دعم عملاء محترف في شركة NileTel للاتصالات.

قواعد صارمة:
- أجب باللهجة المصرية الطبيعية وبأدب.
- استخدم فقط المعلومات الموجودة في السياق. ممنوع التأليف أو اختراع تفاصيل غير موجودة.
- إذا كان المستخدم يطلب إنشاء تذكرة أو تصعيد أو إرسال مهندس، اجعل needs_action: YES.
- إذا كان السؤال استفسار معلوماتي فقط، اجعل needs_action: NO.
- لا تخترع أرقام تذاكر أو خطوات غير مذكورة في السياق.
- التزم بهذا الشكل فقط وبدون أي سطور إضافية:
answer: <your answer>
needs_action: YES أو NO"""

        user_prompt = f"""السياق المتاح:
{context}

السؤال:
{query}"""


        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200
        )

        text = response.choices[0].message.content or ""
        text = text.strip()

        # remove repetition
        text = self.remove_repetition(text)

        # extract decision
        clean_answer, needs_action = self.parse_llm_response(text)

        best = max(retrieved_results, key=lambda x: x["score"])

        return {
            "answer": clean_answer,
            "needs_action": needs_action,
            "sources": [r["source"] for r in retrieved_results],
            "displayed_source": best["source"]
        }

    # ============================================================
    # RUN RAG PIPELINE
    # ============================================================
    def run_rag_pipeline(self, query):
        print(f"\n{'=' * 80}")
        print(f"User Query: {query}")

        route = self.route_query(query)
        print(f"Route selected: {route}")

        # Handle casual conversation without LLM
        if route == "casual_conv":
            return {
                "answer": "أهلاً يا فندم 😊، تحت أمرك في أي استفسار عن خدمات NileTel.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "General"
            }

        if route == "out_of_scope":
            return {
                "answer": "السؤال ده خارج نطاق خدمة NileTel يا فندم.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "Unknown"
            }

        if route == "ticket":
            return {
                "answer": "تمام يا فندم، هعمل تذكرة فوراً.",
                "needs_action": "YES",
                "sources": [],
                "displayed_source": "Ticket System"
            }

        # Handle inquiry with RAG + LLM
        results = self.retrieve(query)
        return self.generate_answer(query, results)


# ================== MAIN ==================
if __name__ == "__main__":
    rag = TelecomRAG()

    queries = [
        "ازاي أحل مشكلة 5G throttling؟",
        "النت مقطوع اعمل تذكرة",
        "ازيك"
    ]

    for q in queries:
        res = rag.run_rag_pipeline(q)
        print(res)
        print("-" * 60)