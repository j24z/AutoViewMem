#!/usr/bin/env python3
import json
import os
import sys
import time
import math
import threading
import traceback
import re
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Template
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Load environment variables from .env
load_dotenv()

# Import configuration
from autoviewmem.config import COLLECTION_NAME, DATA_TEST_PATH, EMBEDDING_NAME, EMBEDDING_SIZE, LLM_TEMPERATURE, LLM_MAX_TOKENS, QDRANT_URL
# Import answer prompt template
from autoviewmem.evaluation.prompts import ANSWER_PROMPT
# Import logger
from autoviewmem.memory.log_config import logger

# Configuration
MODEL_NAME = os.getenv("ARK_MODEL", "qwen3-8b")
OPENAI_BASE_URL = os.getenv("ARK_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("ARK_API_KEY", "any")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "empty")

def _guess_thread_source(thread_name: str, stack: str) -> str:
    lowered = (thread_name + "\n" + stack).lower()
    if "posthog" in lowered or "mem0.memory.telemetry" in lowered:
        return "mem0 telemetry / posthog"
    if "grpc" in lowered or "cygrpc" in lowered:
        return "grpc (likely qdrant-client)"
    if "qdrant" in lowered:
        return "qdrant-client"
    if "httpx" in lowered or "httpcore" in lowered:
        return "httpx/httpcore"
    if "openai" in lowered:
        return "openai (httpx)"
    if "concurrent.futures" in lowered or "threadpoolexecutor" in lowered:
        return "ThreadPoolExecutor"
    return "unknown"


def dump_live_threads(tag: str) -> None:
    frames = sys._current_frames()
    threads = list(threading.enumerate())
    logger.info(f"\n=== live threads ({tag}) count={len(threads)} ===")
    for t in threads:
        frame = frames.get(t.ident)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else ""
        source = _guess_thread_source(t.name, stack)
        logger.info(f"- name={t.name} daemon={t.daemon} ident={t.ident} source={source}")
        if not t.daemon and t.name != "MainThread":
            lines = stack.splitlines()
            tail = "\n".join(lines[-12:]) if lines else ""
            if tail:
                logger.info(tail)



class BM25Index:
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_corpus)
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        df = defaultdict(int)
        postings = defaultdict(list)
        for i, doc in enumerate(tokenized_corpus):
            if not doc:
                continue
            tf = Counter(doc)
            for term, freq in tf.items():
                df[term] += 1
                postings[term].append((i, freq))

        self.postings = postings
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def score(self, tokenized_query):
        if self.n_docs == 0 or self.avgdl <= 0:
            return {}

        scores = defaultdict(float)
        for term in tokenized_query:
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for doc_idx, tf in self.postings.get(term, []):
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += term_idf * (tf * (self.k1 + 1.0) / denom)
        return dict(scores)


class MultiLayerMemorySearch:
    def __init__(self, output_path=None, top_k=15, routing_mode="llm", fixed_limits=None, l1_collection=COLLECTION_NAME, n_adaptive_views=1, l1_view_limit=None, enable_view_routing=True, enable_cache=True, max_cache_users=1024, verbose=False):
        self.top_k = top_k
        self.verbose = verbose
        # Note: routing_mode, fixed_limits, n_adaptive_views, enable_view_routing are ignored in this version
        # as we are enforcing a hybrid search on the "L2" (l1_collection) layer only.
        
        self.collection_name = l1_collection
        self.enable_cache = bool(enable_cache)
        self.max_cache_users = int(max_cache_users) if max_cache_users is not None else None
        self.enable_planning = True
        self.enable_reflection = True
        self.enable_parallel_retrieval = True
        self.max_reflection_rounds = 2
        self.min_planning_queries = 3
        self.max_planning_queries = 5
        self.max_retrieval_workers = 4
        
        # Initialize Clients
        self.openai_client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
        
        self.embedding_client = OpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
        )
        
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        except TypeError:
            self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        self.results = defaultdict(list)
        self.output_path = output_path if output_path else self._generate_output_path()
        self.routing_log_path = self.output_path.replace(".json", "_routing_log.json")
        self.retrieval_log_path = (
            self.output_path.replace(".json", "_retrieval_log.jsonl")
            if self.output_path.endswith(".json")
            else (self.output_path + "_retrieval_log.jsonl")
        )
        self.routing_logs = []
        self.file_lock = threading.Lock()
        self._user_cache_lock = threading.Lock()
        self._user_cache = OrderedDict()
        self._user_cache_events = {}
        
        logger.info(f"Using model: {MODEL_NAME}")
        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        logger.info(f"Initialized Hybrid Search on collection: {self.collection_name}")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _generate_output_path(self):
        """Generate output filename with timestamp, model and collection info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Shorten model name
        short_model = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
        short_model = short_model.replace(":", "_")
        
        filename = f"routing_{short_model}_{self.collection_name}_{timestamp}.json"
        
        output_dir = os.path.join("outputs", date_str)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with self.file_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _get_candidate_payload(self, id_to_record: Dict[Any, Any], pid: Any) -> Dict[str, Any]:
        rec = id_to_record.get(pid)
        if rec is None:
            return {}
        return getattr(rec, "payload", None) or {}

    def _log_hybrid_search_trace(
        self,
        query: str,
        user_id: str,
        limit: int,
        candidate_pool_limit: int,
        bm25_ranked: List[Tuple[Any, float]],
        vector_results: List[Any],
        sorted_ids: List[Tuple[Any, float]],
        bm25_rank_by_id: Dict[Any, int],
        bm25_score_by_id: Dict[Any, float],
        vector_rank_by_id: Dict[Any, int],
        vector_score_by_id: Dict[Any, Optional[float]],
        id_to_record: Dict[Any, Any],
        k: int,
    ) -> None:
        ts = datetime.now().isoformat()

        bm25_candidates = []
        for rank, (pid, score) in enumerate(bm25_ranked[:candidate_pool_limit]):
            payload = self._get_candidate_payload(id_to_record, pid)
            bm25_candidates.append(
                {
                    "entry_id": str(pid),
                    "rank": int(rank),
                    "score": float(score),
                    "memory": self._payload_to_text(payload),
                    "timestamp": payload.get("timestamp", "") or payload.get("created_at", ""),
                    "view": payload.get("view", None),
                }
            )

        embedding_candidates = []
        for rank, res in enumerate(vector_results[:candidate_pool_limit]):
            pid = getattr(res, "id", None)
            if pid is None:
                continue
            payload = self._get_candidate_payload(id_to_record, pid)
            embedding_candidates.append(
                {
                    "entry_id": str(pid),
                    "rank": int(rank),
                    "score": vector_score_by_id.get(pid),
                    "memory": self._payload_to_text(payload),
                    "timestamp": payload.get("timestamp", "") or payload.get("created_at", ""),
                    "view": payload.get("view", None),
                }
            )

        fused_results = []
        for rank, (pid, rrf_score) in enumerate(sorted_ids):
            if pid is None:
                continue
            payload = self._get_candidate_payload(id_to_record, pid)
            fused_results.append(
                {
                    "entry_id": str(pid),
                    "rank": int(rank),
                    "rrf_score": float(rrf_score),
                    "bm25_rank": bm25_rank_by_id.get(pid),
                    "bm25_score": bm25_score_by_id.get(pid),
                    "embedding_rank": vector_rank_by_id.get(pid),
                    "embedding_score": vector_score_by_id.get(pid),
                    "memory": self._payload_to_text(payload),
                    "timestamp": payload.get("timestamp", "") or payload.get("created_at", ""),
                    "view": payload.get("view", None),
                }
            )

        record = {
            "ts": ts,
            "query": query,
            "user_id": str(user_id),
            "limit": int(limit),
            "candidate_pool_limit": int(candidate_pool_limit),
            "rrf_k": int(k),
            "bm25": {
                "count": int(len(bm25_ranked)),
                "candidates": bm25_candidates,
            },
            "embedding": {
                "count": int(len(vector_results)),
                "candidates": embedding_candidates,
            },
            "fused": {
                "count": int(len(sorted_ids)),
                "results": fused_results,
            },
        }
        self._append_jsonl(self.retrieval_log_path, record)

    def _get_embedding(self, text):
        try:
            text = text.replace("\n", " ")
            return self.embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * EMBEDDING_SIZE

    def _tokenize(self, text):
        if not text:
            return []
        text = text.lower()
        # Strategy:
        # 1. Match English words/numbers: [a-z0-9]+
        # 2. Match Chinese characters (and other non-ascii): [\u4e00-\u9fa5] (Basic CJK)
        #    Actually, splitting by character for CJK is a simple robust baseline.
        
        tokens = []
        # Find all English words/numbers
        tokens.extend(re.findall(r'[a-z0-9]+', text))
        
        # Find all CJK characters (treat each as a token)
        # This range covers most common CJK Unified Ideographs
        tokens.extend(re.findall(r'[\u4e00-\u9fa5]', text))
        
        return tokens

    def _sanitize_query_for_user(self, query: str, user_id: str) -> str:
        q = (query or "").strip()
        uid = (user_id or "").strip()
        if not q or not uid:
            return q

        base = re.sub(r"_\d+$", "", uid)
        aliases = [uid, base]

        cleaned = q
        for a in sorted({x for x in aliases if x}, key=len, reverse=True):
            if re.fullmatch(r"[A-Za-z0-9_]+", a):
                cleaned = re.sub(rf"\b{re.escape(a)}\b[:：]?\s*", "", cleaned)
            else:
                cleaned = cleaned.replace(a, "")

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[，,。.\-—:：\s]+", "", cleaned).strip()
        return cleaned or q

    def _payload_to_text(self, payload):
        if not payload:
            return ""
        for key in ("memory", "data", "text", "content", "message"):
            v = payload.get(key)
            if v is None:
                continue
            if isinstance(v, str):
                if v.strip():
                    return v
                continue
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return ""

    def _build_user_cache_entry(self, user_id):
        all_records = []
        next_page_offset = None
        while True:
            records, next_page_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=1000,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            all_records.extend(records)
            if next_page_offset is None:
                break

        id_to_record = {rec.id: rec for rec in all_records}
        corpus = [self._payload_to_text(getattr(rec, "payload", None) or {}) for rec in all_records]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        bm25_index = BM25Index(tokenized_corpus) if tokenized_corpus else None

        return {
            "all_records": all_records,
            "id_to_record": id_to_record,
            "bm25_index": bm25_index,
        }

    def _get_user_cache_entry(self, user_id):
        if not self.enable_cache:
            return self._build_user_cache_entry(user_id)

        with self._user_cache_lock:
            entry = self._user_cache.get(user_id)
            if entry is not None:
                self._user_cache.move_to_end(user_id)
                return entry

            evt = self._user_cache_events.get(user_id)
            if evt is None:
                evt = threading.Event()
                self._user_cache_events[user_id] = evt
                creator = True
            else:
                creator = False

        if not creator:
            evt.wait()
            with self._user_cache_lock:
                entry = self._user_cache.get(user_id)
                if entry is not None:
                    self._user_cache.move_to_end(user_id)
                    return entry
            return self._get_user_cache_entry(user_id)

        try:
            entry = self._build_user_cache_entry(user_id)
            with self._user_cache_lock:
                self._user_cache[user_id] = entry
                self._user_cache.move_to_end(user_id)
                if self.max_cache_users is not None:
                    while len(self._user_cache) > self.max_cache_users:
                        self._user_cache.popitem(last=False)
            return entry
        finally:
            with self._user_cache_lock:
                evt = self._user_cache_events.pop(user_id, None)
                if evt is not None:
                    evt.set()

    def hybrid_search(self, query, user_id, limit):
        """
        Perform hybrid search (Vector + BM25) on Qdrant collection.
        """
        try:
            entry = self._get_user_cache_entry(user_id)
            all_records = entry["all_records"]
            id_to_record = entry["id_to_record"]
            bm25_index = entry["bm25_index"]
            
            # 2. BM25 Search
            bm25_rank_by_id = {}
            bm25_score_by_id = {}
            bm25_ranked = []
            if bm25_index is not None and all_records:
                tokenized_query = self._tokenize(query)
                idx_scores = bm25_index.score(tokenized_query)

                bm25_ranked = [(all_records[idx].id, float(score)) for idx, score in idx_scores.items() if score > 0]
                bm25_ranked.sort(key=lambda x: x[1], reverse=True)
                
                # Create rank map: id -> rank (0-based)
                for rank, (pid, score) in enumerate(bm25_ranked):
                    bm25_rank_by_id[pid] = rank
                    bm25_score_by_id[pid] = score
            
            # 3. Vector Search
            query_vector = self._get_embedding(query)
            candidate_pool_limit = max(limit * 5, 50)
            vector_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=candidate_pool_limit
            )
            vector_results = list(getattr(vector_response, "points", []) or [])
            vector_rank_by_id = {}
            vector_score_by_id = {}
            for idx, res in enumerate(vector_results):
                pid = getattr(res, "id", None)
                if pid is None:
                    continue
                vector_rank_by_id[pid] = idx
                score_val = getattr(res, "score", None)
                vector_score_by_id[pid] = float(score_val) if score_val is not None else None
            
            # 4. RRF Fusion
            # RRF score = 1 / (k + rank)
            k = 60
            final_scores = defaultdict(float)
            
            # Collect all unique IDs involved
            all_ids = set(bm25_rank_by_id.keys()) | set(vector_rank_by_id.keys())
            
            for pid in all_ids:
                if pid is None:
                    continue
                r_bm25 = bm25_rank_by_id.get(pid, 10000)
                r_vec = vector_rank_by_id.get(pid, 10000)
                
                score = 0
                if pid in bm25_rank_by_id:
                    score += 1 / (k + r_bm25)
                if pid in vector_rank_by_id:
                    score += 1 / (k + r_vec)
                final_scores[pid] = score
            
            # Sort by RRF score
            sorted_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            # Retrieve final records
            final_memories = []
            for pid, score in sorted_ids:
                if pid in id_to_record:
                    rec = id_to_record[pid]
                    payload = getattr(rec, "payload", None) or {}
                    final_memories.append({
                        "entry_id": str(pid),
                        "memory": self._payload_to_text(payload),
                        "timestamp": payload.get("timestamp", "") or payload.get("created_at", ""),
                        "score": round(score, 4), # RRF score
                        "view": payload.get("view", None) # If available
                    })
                # If ID was from vector search but not in scroll (unlikely unless concurrent delete), fetch it?
                # Since we scrolled ALL records first, id_to_record should have it.
            
            try:
                self._log_hybrid_search_trace(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    candidate_pool_limit=candidate_pool_limit,
                    bm25_ranked=bm25_ranked,
                    vector_results=vector_results,
                    sorted_ids=sorted_ids,
                    bm25_rank_by_id=bm25_rank_by_id,
                    bm25_score_by_id=bm25_score_by_id,
                    vector_rank_by_id=vector_rank_by_id,
                    vector_score_by_id=vector_score_by_id,
                    id_to_record=id_to_record,
                    k=k,
                )
            except Exception as _e:
                if self.verbose:
                    print(f"[TRACE] Failed to write retrieval trace: {_e}")

            return final_memories, {"bm25_count": len(bm25_rank_by_id), "vector_count": len(vector_results)}

        except Exception as e:
            print(f"Hybrid Search Error: {e}")
            traceback.print_exc()
            return [], {}

    def _chat_json(self, system: str, user: str, json_schema: Dict[str, Any], temperature: float = 0.2) -> Dict[str, Any]:
        response = self._chat_completion_with_retry(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
            guided_json_schema=json_schema,
        )
        if not response or not response.choices:
            return {}
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        parsed = self._extract_json(content)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _analyze_information_requirements(self, question: str) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n[TRACE] _analyze_information_requirements input: {question}")
        schema = {
            "type": "object",
            "properties": {
                "question_type": {"type": "string"},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "required_info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "info_type": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string"},
                        },
                        "required": ["info_type", "description", "priority"],
                        "additionalProperties": True,
                    },
                },
                "relationships": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question_type", "key_entities", "required_info", "relationships"],
            "additionalProperties": True,
        }
        prompt = f"""
Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {question}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2", ...],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2", ...],
  "minimal_queries_needed": 2
}}
```

Focus on identifying the minimal essential information needed, not exhaustive details.

Return ONLY the JSON, no other text.
"""
        result = self._chat_json(
            system="You are an intelligent information requirement analyst. You must output valid JSON format.",
            user=prompt,
            json_schema=schema,
            temperature=0.2,
        )
        if self.verbose:
            print(f"[TRACE] _analyze_information_requirements output: {json.dumps(self._extract_json(json.dumps(result)) if isinstance(result, dict) else result, ensure_ascii=False, indent=2)}")
        return result

    def _generate_targeted_queries(self, question: str, analysis: Dict[str, Any]) -> List[str]:
        if self.verbose:
            print(f"\n[TRACE] _generate_targeted_queries input question: {question}")
            print(f"[TRACE] _generate_targeted_queries input analysis: {json.dumps(analysis, ensure_ascii=False)}")
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries"],
            "additionalProperties": True,
        }
        prompt = f"""
Based on the information requirements analysis, generate 3-5 targeted search queries needed to gather the required information.

Original Question: {question}

Information Requirements Analysis:
- Question Type: {analysis.get('question_type', 'general')}
- Key Entities: {analysis.get('key_entities', [])}
- Required Information: {analysis.get('required_info', [])}
- Relationships: {analysis.get('relationships', [])}
- Minimal Queries Needed: {analysis.get('minimal_queries_needed', 1)}

Generate 3-5 search queries that would efficiently gather all the required information. Each query should be focused and specific to retrieve distinct types of information.

Guidelines:
1. Always include the original query as one option
2. Generate 3-5 queries
3. Each query should target a specific information requirement or missing facet
4. Avoid redundant or overlapping queries
5. Use concise phrasing; avoid long sentences

Return your response in JSON format:
```json
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": [
    "targeted query 1",
    "targeted query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._chat_json(
            system="You are a query generation specialist. You must output valid JSON format.",
            user=prompt,
            json_schema=schema,
            temperature=0.3,
        )
        queries = result.get("queries", []) or []
        normalized = []
        seen = set()
        for q in [question] + queries:
            s = (q or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            normalized.append(s)
        if len(normalized) < self.min_planning_queries:
            key_entities = analysis.get("key_entities", [])
            if not isinstance(key_entities, list):
                key_entities = []
            for ent in key_entities[: self.min_planning_queries * 2]:
                s = f"{ent} {question}".strip()
                if s not in seen:
                    seen.add(s)
                    normalized.append(s)
                if len(normalized) >= self.min_planning_queries:
                    break
        final_queries = normalized[: self.max_planning_queries]
        if self.verbose:
            print(f"[TRACE] _generate_targeted_queries output: {final_queries}")
        return final_queries

    def _semantic_search(self, query: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        memories, _meta = self.hybrid_search(query, user_id, limit)
        return memories

    def _merge_and_deduplicate_entries(self, entries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        best_by_id: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            eid = str(e.get("entry_id") or "").strip()
            if not eid:
                continue
            score = e.get("score", 0.0)
            prev = best_by_id.get(eid)
            if prev is None or (isinstance(score, (int, float)) and score > prev.get("score", 0.0)):
                best_by_id[eid] = e
        merged = list(best_by_id.values())
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:limit]

    def _check_answer_adequacy(self, question: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n[TRACE] _check_answer_adequacy input question: {question}")
            print(f"[TRACE] _check_answer_adequacy input memories count: {len(memories)}")
        schema = {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "reasoning": {"type": "string"},
                "missing_info": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["assessment", "missing_info"],
            "additionalProperties": True,
        }
        context = []
        for i, m in enumerate(memories[: min(12, len(memories))], 1):
            ts = (m.get("timestamp") or "").strip()
            mem = (m.get("memory") or "").strip()
            if not mem:
                continue
            if ts:
                context.append(f"[{i}] {ts}: {mem}")
            else:
                context.append(f"[{i}] {mem}")
        prompt = f"""
You are evaluating whether the provided context contains sufficient information to answer a user question.

Question: {question}

Context:
{chr(10).join(context)}

Please evaluate whether the context contains enough information to provide a meaningful, accurate answer to the question.

Consider these criteria:
1. Does the context directly address the question being asked?
2. Are there key details necessary to answer the question?
3. Is the information specific enough to avoid vague responses?

Return your evaluation in JSON format:
```json
{{
  "assessment": "Sufficient" OR "Insufficient",
  "reasoning": "Brief explanation of why the context is or isn't sufficient",
  "missing_info": ["list", "of", "missing", "information"] (only if insufficient)
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._chat_json(
            system="You are an information adequacy evaluator. You must output valid JSON format.",
            user=prompt,
            json_schema=schema,
            temperature=0.1,
        )
        assessment = (result.get("assessment") or "").strip()
        if assessment not in ("Sufficient", "Insufficient"):
            result["assessment"] = "Insufficient"
        missing = result.get("missing_info")
        if not isinstance(missing, list):
            result["missing_info"] = []
        if self.verbose:
            print(f"[TRACE] _check_answer_adequacy output: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return result

    def _generate_additional_queries(self, question: str, missing_info: List[str], memories: List[Dict[str, Any]]) -> List[str]:
        if self.verbose:
            print(f"\n[TRACE] _generate_additional_queries input question: {question}")
            print(f"[TRACE] _generate_additional_queries input missing_info: {missing_info}")
        schema = {
            "type": "object",
            "properties": {
                "additional_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["additional_queries"],
            "additionalProperties": True,
        }
        context = []
        for i, m in enumerate(memories[: min(8, len(memories))], 1):
            mem = (m.get("memory") or "").strip()
            if mem:
                context.append(f"[{i}] {mem}")
        prompt = f"""
Based on the original question and current available information, generate additional specific search queries that would help find the missing information needed to answer the question completely.

Original Question: {question}

Missing Information Points:
{missing_info}

Current Available Information:
{chr(10).join(context)}

Analyze what specific information is still missing and generate 2-4 targeted search queries that would help find this missing information.

The queries should be:
1. Specific and focused on the missing information
2. Different from the original question
3. Likely to find complementary information

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what's missing",
  "additional_queries": [
    "specific search query 1",
    "specific search query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._chat_json(
            system="You are a search strategy assistant. You must output valid JSON format.",
            user=prompt,
            json_schema=schema,
            temperature=0.3,
        )
        queries = result.get("additional_queries", [])
        if not isinstance(queries, list):
            queries = []
        out = []
        seen = set()
        for q in queries:
            s = (q or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        final_queries = out[:3]
        if self.verbose:
            print(f"[TRACE] _generate_additional_queries output: {final_queries}")
        return final_queries

    def retrieve(self, question: str, user_id: str, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        limit = int(limit or self.top_k)
        search_question = self._sanitize_query_for_user(question, user_id)
        if not self.enable_planning:
            memories = self._semantic_search(search_question, user_id, limit)
            return self._merge_and_deduplicate_entries(memories, limit), {"mode": "no_planning"}

        analysis = self._analyze_information_requirements(search_question)
        queries = self._generate_targeted_queries(search_question, analysis)
        queries = [self._sanitize_query_for_user(q, user_id) for q in (queries or [])]

        all_entries: List[Dict[str, Any]] = []
        if self.enable_parallel_retrieval and len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_retrieval_workers, len(queries))) as executor:
                futures = {executor.submit(self._semantic_search, q, user_id, limit): q for q in queries}
                for fut in as_completed(futures):
                    try:
                        all_entries.extend(fut.result() or [])
                    except Exception:
                        traceback.print_exc()
        else:
            for q in queries:
                all_entries.extend(self._semantic_search(q, user_id, limit) or [])

        merged = self._merge_and_deduplicate_entries(all_entries, limit)

        if self.enable_reflection:
            for _round in range(self.max_reflection_rounds):
                adequacy = self._check_answer_adequacy(question, merged)
                if adequacy.get("assessment") == "Sufficient":
                    break
                missing_info = adequacy.get("missing_info", []) or []
                additional = self._generate_additional_queries(search_question, missing_info, merged)
                additional = [self._sanitize_query_for_user(q, user_id) for q in (additional or [])]
                if not additional:
                    break
                additional_entries: List[Dict[str, Any]] = []
                if self.enable_parallel_retrieval and len(additional) > 1:
                    with ThreadPoolExecutor(max_workers=min(self.max_retrieval_workers, len(additional))) as executor:
                        futures = {executor.submit(self._semantic_search, q, user_id, limit): q for q in additional}
                        for fut in as_completed(futures):
                            try:
                                additional_entries.extend(fut.result() or [])
                            except Exception:
                                traceback.print_exc()
                else:
                    for q in additional:
                        additional_entries.extend(self._semantic_search(q, user_id, limit) or [])
                merged = self._merge_and_deduplicate_entries(merged + additional_entries, limit)

        return merged, {"mode": "planning_reflection", "queries": queries}

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question):
        """Answer question using hybrid search"""
        
        s1_memories, s1_meta = self.retrieve(question, speaker_1_user_id, self.top_k)
        s2_memories, s2_meta = self.retrieve(question, speaker_2_user_id, self.top_k)
            
        # Format memories
        def format_memories(memories):
            formatted = []
            for mem in memories:
                formatted.append(f"[{mem.get('timestamp', 'N/A')}]: {mem['memory']}")
            return formatted

        speaker_1_txt = format_memories(s1_memories)
        speaker_2_txt = format_memories(s2_memories)
        
        # Combine context
        context_parts = []
        if speaker_1_txt:
            context_parts.append(f"User {speaker_1_user_id} Memories:\n" + "\n".join(speaker_1_txt))
        if speaker_2_txt:
            context_parts.append(f"User {speaker_2_user_id} Memories:\n" + "\n".join(speaker_2_txt))
        context_str = "\n\n".join(context_parts)
        
        prompt = f"""Answer the user's question based on the provided context. 
 
 User Question: {question} 
 
 Relevant Context: 
 {context_str} 
 
 Requirements: 
 1. First, think through the reasoning process 
 2. Then provide a very CONCISE answer (short phrase about core information) 
 3. Answer must be based ONLY on the provided context 
 4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed 
 5. Return your response in JSON format 
 
 Output Format: 
 {{ 
   "reasoning": "Brief explanation of your thought process", 
   "answer": "Concise answer in a short phrase" 
 }} 
 ... 
 Now answer the question. Return ONLY the JSON, no other text."""

        print(f"Prompt Length: {len(prompt)}")
        #print(prompt)
        try:
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["reasoning", "answer"]
            }
            
            response = self._chat_completion_with_retry(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                guided_json_schema=schema
            )

            content = ""
            reasoning_content = ""

            if response and response.choices:
                msg = response.choices[0].message
                raw_content = msg.content or ""
                
                # Try to parse JSON
                parsed = self._extract_json(raw_content)
                if parsed:
                    content = parsed.get("answer", "")
                    reasoning_content = parsed.get("reasoning", "")
                else:
                    content = raw_content
                
                # Pack results to match expected format in process_question
                s1_out = {
                    "policy": {"mode": "hybrid"},
                    "limits_used": {"top_k": self.top_k},
                    "results": {"L2": s1_memories}, # Wrap in "L2" key for consistency
                    "l1_routing_details": s1_meta
                }
                s2_out = {
                    "policy": {"mode": "hybrid"},
                    "limits_used": {"top_k": self.top_k},
                    "results": {"L2": s2_memories},
                    "l1_routing_details": s2_meta
                }
                
                return content, reasoning_content, s1_out, s2_out
            else:
                return "I don't know", "", {}, {}
        except Exception as e:
            print(f"Error getting answer: {e}")
            traceback.print_exc()
            return "I don't know", "", {}, {}

    def _chat_completion_with_retry(self, model, messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS, max_retries=10, base_delay=2, max_delay=120, guided_json_schema: Optional[Dict[str, Any]] = None):
        for attempt in range(max_retries):
            try:
                extra_body: Dict[str, Any] = {
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    },
                }
                if guided_json_schema is not None:
                    extra_body["guided_json"] = guided_json_schema
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body
                )
                return response
            except RateLimitError as e:
                wait = min(base_delay * (2 ** attempt), max_delay)
                print(f"Rate limit hit, waiting {wait}s")
                time.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1: raise
                time.sleep(1)

    def process_data_file(self, file_path, max_workers=None):
        """Process dataset file"""
        print(f"Loading dataset from {file_path}...")
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        for idx, item in enumerate(data):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in qa:
                tasks.append({
                    "val": question_item,
                    "speaker_a": speaker_a_user_id,
                    "speaker_b": speaker_b_user_id,
                    "idx": idx
                })

        print(f"Total questions to process: {len(tasks)}")

        with ThreadPoolExecutor(max_workers=64) as executor:
            future_to_task = {
                executor.submit(
                    self.process_question,
                    task["val"],
                    task["speaker_a"],
                    task["speaker_b"],
                    task["idx"]
                ): task for task in tasks
            }

            for i, future in tqdm(enumerate(as_completed(future_to_task)), total=len(tasks), desc="Processing All Questions"):
                try:
                    conversation_idx, result, logs = future.result()
                    # Print progress for external monitoring
                    print(f"Processing question {i+1}/{len(tasks)}", flush=True)

                    with self.file_lock:
                        self.results[conversation_idx].append(result)
                        self.routing_logs.extend(logs)

                    if i % 100 == 0:
                        self.save_results()
                        self.save_routing_logs()

                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    traceback.print_exc()

        self.save_results()
        self.save_routing_logs()
        print(f"Processing complete. Results saved to {self.output_path}")
        print(f"Routing logs saved to {self.routing_log_path}")

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, conversation_idx):
        """Process single question"""
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        start_time = time.time()
        response, reasoning_content, s1_out, s2_out = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)
        end_time = time.time()
        
        # Safe extraction even if error occurred
        s1_results = s1_out.get("results", {})
        s2_results = s2_out.get("results", {})

        # Format retrieved memories for storage
        def format_results_for_storage(results):
            formatted = []
            for layer, memories in results.items():
                for mem in memories:
                    mem_data = {
                        "layer": layer,
                        "memory": mem['memory'],
                        "score": mem['score'],
                        "timestamp": mem['timestamp']
                    }
                    if "view" in mem:
                        mem_data["view"] = mem["view"]
                    formatted.append(mem_data)
            return formatted

        speaker_1_formatted = format_results_for_storage(s1_results)
        speaker_2_formatted = format_results_for_storage(s2_results)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "reasoning_content": reasoning_content,
            "adversarial_answer": adversarial_answer,
            "processing_time": end_time - start_time,
            "speaker_1_memories": speaker_1_formatted,
            "speaker_2_memories": speaker_2_formatted,
            "num_speaker_1_memories": len(speaker_1_formatted),
            "num_speaker_2_memories": len(speaker_2_formatted),
            "speaker_1_policy": s1_out.get("policy", {}),
            "speaker_2_policy": s2_out.get("policy", {}),
            "speaker_1_limits": s1_out.get("limits_used", {}),
            "speaker_2_limits": s2_out.get("limits_used", {})
        }
        
        logs = []
        if "l1_routing_details" in s1_out and s1_out["l1_routing_details"]:
             logs.append({
                 "conversation_idx": conversation_idx,
                 "speaker": speaker_a_user_id,
                 "details": s1_out["l1_routing_details"]
             })
        if "l1_routing_details" in s2_out and s2_out["l1_routing_details"]:
             logs.append({
                 "conversation_idx": conversation_idx,
                 "speaker": speaker_b_user_id,
                 "details": s2_out["l1_routing_details"]
             })

        return conversation_idx, result, logs

    def save_results(self):
        """Save results to file"""
        with self.file_lock:
            with open(self.output_path, "w", encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

    def save_routing_logs(self):
        """Save routing logs to file"""
        with self.file_lock:
            with open(self.routing_log_path, "w", encoding='utf-8') as f:
                json.dump(self.routing_logs, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

    def close(self):
        client = getattr(self, "openai_client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as e:
                print(f"Error closing OpenAI client: {e}")
                
        embed_client = getattr(self, "embedding_client", None)
        if embed_client is not None and hasattr(embed_client, "close"):
            try:
                embed_client.close()
            except Exception as e:
                print(f"Error closing Embedding client: {e}")
                
        q_client = getattr(self, "qdrant_client", None)
        if q_client is not None:
            # QdrantClient (sync) doesn't strictly need close, but good practice if available
            try:
                 q_client.close()
            except:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid memory search (Vector + BM25) on Qdrant")
    parser.add_argument("--input", default=DATA_TEST_PATH, help="Input dataset path")
    parser.add_argument("--output", help="Output results path")
    parser.add_argument("--top_k", type=int, default=30, help="Number of memories to retrieve per speaker")
    parser.add_argument("--l1_collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--test_question", help="Run a single test question for debugging")
    parser.add_argument("--speaker_a", default="Melanie_0", help="Speaker A ID for test")
    parser.add_argument("--speaker_b", default="Caroline_0", help="Speaker B ID for test")
    args = parser.parse_args()

    if args.output:
        date_str = time.strftime("%Y%m%d")
        output_dir = os.path.dirname(args.output)
        output_filename = os.path.basename(args.output)
        new_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(new_output_dir, exist_ok=True)
        args.output = os.path.join(new_output_dir, output_filename)
        print(f"Output will be saved to: {args.output}")

    if args.test_question:
        print(f"Running single test question: {args.test_question}")
        searcher = MultiLayerMemorySearch(
            output_path=args.output,
            top_k=args.top_k,
            l1_collection=args.l1_collection,
            verbose=True
        )
        try:
            res, reasoning, s1, s2 = searcher.answer_question(args.speaker_a, args.speaker_b, args.test_question)
            print("\n=== Final Answer ===")
            print(res)
            print("\n=== Reasoning ===")
            print(reasoning)
        finally:
            if searcher:
                searcher.close()
    else:
        print("Starting multi-layer memory search (Hybrid L2 Only)...")
        searcher = None
        try:
            searcher = MultiLayerMemorySearch(
                output_path=args.output, 
                top_k=args.top_k,
                l1_collection=args.l1_collection,
            )
            searcher.process_data_file(args.input)
        finally:
            if searcher is not None:
                try:
                    searcher.save_results()
                except Exception:
                    pass
                dump_live_threads("before_close")
                try:
                    searcher.close()
                finally:
                    dump_live_threads("after_close")
