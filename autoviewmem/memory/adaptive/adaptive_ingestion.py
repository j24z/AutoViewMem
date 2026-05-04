import json
import os
import time
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from mem0 import Memory
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, EMBEDDING_NAME, EMBEDDING_SIZE, EMBEDDING_BASE_URL, EMBEDDING_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS, QDRANT_URL
from .schema_discovery import SchemaDiscovery
from .utils import call_llm_with_retry
from collections import defaultdict
from ..log_config import logger

# Load environment variables
load_dotenv()

class AdaptiveMemoryLayer:
    def __init__(self, warm_start_steps=50, k_views=3, debug=False, collection_name="adaptive_memory_layer_1_24", use_dpp=False, llm_param_format="vllm", enable_thinking=False):
        self.warm_start_steps = warm_start_steps
        self.k_views = k_views
        self.debug = debug
        self.collection_name = collection_name
        self.is_frozen = False
        self.use_dpp = use_dpp
        self.llm_param_format = llm_param_format
        self.enable_thinking = enable_thinking
        self.json_failure_count = 0
        
        # User-specific data structures
        self.user_active_prompts = {} # {user_id: [prompts]}
        self.user_active_prompt_embeddings = {} # {user_id: embeddings}
        self.user_candidate_pools = defaultdict(list) # {user_id: [candidates]}
        self.user_candidate_pools_observed = defaultdict(list)
        
        # Buffer stores tuples of (messages, user_id, metadata) per user
        self.user_buffers = defaultdict(list)
        self.user_buffer_turn_counts = defaultdict(int)
        
        self.schema_discovery = SchemaDiscovery(llm_param_format=llm_param_format, enable_thinking=enable_thinking)
        
        qdrant_url = QDRANT_URL.rstrip("/")

        # Initialize mem0 for storage
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "url": qdrant_url,
                    "embedding_model_dims": EMBEDDING_SIZE
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": ARK_MODEL, 
                    "openai_base_url": ARK_BASE_URL, 
                    "api_key": ARK_API_KEY,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    }
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": EMBEDDING_NAME,
                    "openai_base_url": EMBEDDING_BASE_URL,
                    "api_key": EMBEDDING_API_KEY,
                    "embedding_dims": EMBEDDING_SIZE
                },
            },
            }
        self.mem0 = Memory.from_config(config)
        
        # Initialize LLM client and model reference from SchemaDiscovery/Config
        self.llm_client = self.schema_discovery.llm_client
        self.model = ARK_MODEL

    def _build_llm_create_kwargs(self, json_schema=None, schema_name="response"):
        if self.llm_param_format == "openai":
            if json_schema is None:
                return {
                    "extra_body": {
                        "reasoning_effort": "none"
                    }
                }
            schema = json_schema
            if isinstance(schema, dict) and "additionalProperties" not in schema:
                schema = {**schema, "additionalProperties": False}
            return {
                "extra_body": {
                    "reasoning_effort": "none"
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                }
            }

        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                }
            }
        }

    def select_dpp_prompts(self, candidate_prompts, k=3):
        """
        Select K orthogonal prompts using Determinantal Point Process (DPP).
        """
        # Deduplicate
        candidate_prompts = list(set(candidate_prompts))
        n = len(candidate_prompts)
        
        if not candidate_prompts:
            return []
            
        if n <= k:
            return candidate_prompts

        # Step A: Embed
        if not self.schema_discovery.embedder:
            return candidate_prompts[:k]
        
        embeddings = self.schema_discovery.embedder.encode(candidate_prompts)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        
        # Step B: Kernel Matrix
        L = np.dot(embeddings, embeddings.T)
        
        # Step C: Greedy k-DPP
        selected_indices = []
        remaining_indices = list(range(n))
        
        for _ in range(k):
            best_idx = -1
            max_log_det = -float('inf')
            
            for idx in remaining_indices:
                current_indices = selected_indices + [idx]
                L_sub = L[np.ix_(current_indices, current_indices)]
                # Use slogdet to avoid underflow/overflow
                sign, log_det = np.linalg.slogdet(L_sub)
                
                if sign > 0 and log_det > max_log_det:
                    max_log_det = log_det
                    best_idx = idx
                elif len(current_indices) == 1 and best_idx == -1:
                    # Pick the one with largest L_ii (norm squared) if first
                    best_idx = idx
                    max_log_det = 0 
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # Step D: Summarization with Nearest Neighbors
        # For each selected index (centroid), find top 30 nearest neighbors from ALL candidates
        final_prompts = []
        
        def process_dpp_view(idx):
             # Get embedding of the selected view
             centroid_emb = embeddings[idx]
             
             # Calculate cosine similarity with all candidates
             # embeddings is (n, d), centroid_emb is (d,)
             # Since normalized, dot product is cosine similarity
             sims = np.dot(embeddings, centroid_emb)
             
             # Get top 30 indices
             # argsort sorts ascending, so take last 30
             top_n = 30
             if n < top_n:
                 neighbor_indices = np.arange(n)
             else:
                 neighbor_indices = np.argsort(sims)[-top_n:]
             
             # Retrieve texts and similarities
             neighbor_prompts = [candidate_prompts[i] for i in neighbor_indices]
             neighbor_sims = sims[neighbor_indices]
             
             if self.debug:
                 logger.debug(f"DPP View {idx} Similarity Stats (Top {len(neighbor_indices)}):")
                 logger.debug(f"  Range: [{neighbor_sims.min():.4f}, {neighbor_sims.max():.4f}]")
                 logger.debug(f"  Mean: {neighbor_sims.mean():.4f}")
                 # print(f"  Distribution: {np.histogram(neighbor_sims, bins=5)[0]}")
             
             # Summarize
             summary = self.schema_discovery._summarize_cluster_prompts(neighbor_prompts)
             
             # Return both summary and stats
             stats = {
                 "range": [float(neighbor_sims.min()), float(neighbor_sims.max())],
                 "mean": float(neighbor_sims.mean())
             }
             return summary, stats

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=len(selected_indices)) as executor:
            futures = [executor.submit(process_dpp_view, idx) for idx in selected_indices]
            for future in as_completed(futures):
                try:
                    summary, stats = future.result()
                    if summary:
                        # Append tuple of (summary, stats)
                        final_prompts.append({"view": summary, "stats": stats})
                except Exception as e:
                    logger.error(f"Error summarizing DPP view: {e}")

        # If summarization failed for all or returned empty, fallback to raw selected
        if not final_prompts:
             # Fallback needs to match structure: List of dicts or List of strings?
             # To keep it consistent, we wrap in dict with empty stats
             final_prompts = [{"view": candidate_prompts[i], "stats": {}} for i in selected_indices]
        
        # NOTE: Global Library Integration is SKIPPED for DPP as requested.
        return final_prompts

    def reset(self):
        """Clear the vector store collection."""
        try:
            logger.info(f"Resetting collection: {self.collection_name}...")
            self.mem0.reset()
            logger.info("Collection reset complete.")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")

        self.llm_client = self.schema_discovery.llm_client
        self.model = ARK_MODEL
        self.user_candidate_pools_observed = defaultdict(list)

    def ingest_for_discovery(self, messages, user_id, metadata=None):
        """
        Phase 1: Divergence - Accumulate candidates from stream.
        """
        if self.is_frozen:
            return

        self.user_buffers[user_id].append((messages, user_id, metadata))
        self.user_buffer_turn_counts[user_id] += 1
        
        if self.user_buffer_turn_counts[user_id] >= self.warm_start_steps:
            self._generate_candidates_from_buffer(user_id)

    def _generate_candidates_from_buffer(self, user_id):
        full_text = ""
        for msgs, _, _ in self.user_buffers[user_id]:
            for m in msgs:
                full_text += f"{m['role']}: {m['content']}\n"
        
        discovery_text = full_text[-20000:] 
        
        if self.debug:
            logger.debug(f"Prompt extraction input for {user_id}:")
            logger.debug(discovery_text[:500] + "...")
        
        candidates = self.schema_discovery.generate_candidate_prompts(discovery_text)
        if candidates:
            self.user_candidate_pools[user_id].extend(candidates)
            logger.info(f"[Candidate Pool Update] User: {user_id}")
            logger.info(f"    Added {len(candidates)} new candidates.")
            logger.info(f"    Total pool size: {len(self.user_candidate_pools[user_id])}")
            self.user_candidate_pools_observed[user_id].extend(candidates)
            logger.info(f"    Observed pool size: {len(self.user_candidate_pools_observed[user_id])}")
            
            if self.debug:
                logger.debug(f"Extracted candidate prompts for {user_id}:")
                for i, p in enumerate(candidates, start=1):
                    logger.debug(f"{i}. {p}")
        
        self.user_buffers[user_id] = []
        self.user_buffer_turn_counts[user_id] = 0

    def consolidate_discovery(self, user_id=None):
        """
        Phase 2: Convergence - Select K orthogonal prompts.
        If user_id is provided, consolidate only for that user.
        Otherwise, consolidate for all users with candidate pools.
        """
        if user_id:
            users_to_process = [user_id]
        else:
            users_to_process = list(self.user_candidate_pools.keys())
        
        logger.info(f"Consolidating views for {len(users_to_process)} users in parallel...")
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self._consolidate_single_user, uid) for uid in users_to_process]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error consolidating views: {e}")
            
        # Only freeze if this is a global consolidation
        if user_id is None:
            self.is_frozen = True

    def flush_discovery_buffers(self):
        users = list(self.user_buffers.keys())
        for uid in users:
            if self.user_buffers.get(uid):
                self._generate_candidates_from_buffer(uid)
        
    def _consolidate_single_user(self, user_id):
        if not self.user_candidate_pools[user_id]:
            # If buffer has some leftovers, process them
            if self.user_buffers[user_id]:
                self._generate_candidates_from_buffer(user_id)
            
            if not self.user_candidate_pools[user_id]:
                if self.debug:
                    logger.debug(f"No candidates to consolidate for {user_id}.")
                return

        if self.debug:
            logger.debug(f"Consolidating {len(self.user_candidate_pools[user_id])} candidate prompts for {user_id}...")
        
        if self.use_dpp:
            selected = self.select_dpp_prompts(self.user_candidate_pools[user_id], k=self.k_views)
        else:
            selected = self.schema_discovery.select_orthogonal_prompts(self.user_candidate_pools[user_id], k=self.k_views)
        
        # Check if selected items are dicts (from DPP with stats) or strings (from clustering)
        # We need to standardize storage.
        # self.user_active_prompts stores strings (views)
        # We can store stats in a separate dict or change user_active_prompts structure
        # Let's keep user_active_prompts as list of strings for compatibility,
        # and add self.user_view_stats = {} 
        
        final_views = []
        final_stats = []
        
        for item in selected:
            if isinstance(item, dict) and "view" in item:
                final_views.append(item["view"])
                if "stats" in item:
                    final_stats.append(item["stats"])
            elif isinstance(item, str):
                final_views.append(item)
                final_stats.append({}) # Empty stats for clustering method
            else:
                continue

        self.user_active_prompts[user_id] = final_views
        
        # Store stats if we have them (initialize dict if not present)
        if not hasattr(self, "user_view_stats"):
            self.user_view_stats = {}
        self.user_view_stats[user_id] = final_stats
        
        # Cache embeddings for routing
        if self.schema_discovery.embedder:
            self.user_active_prompt_embeddings[user_id] = self.schema_discovery.embedder.encode(final_views)
        
        self.user_candidate_pools[user_id] = [] # Clear
        self.user_buffers[user_id] = [] # Clear any remaining
        
        if self.debug:
            logger.debug(f"Adaptive Layer Frozen for {user_id} with {len(self.user_active_prompts[user_id])} views.")
            for i, p in enumerate(self.user_active_prompts[user_id]):
                stats_str = ""
                if final_stats and i < len(final_stats) and final_stats[i]:
                    s = final_stats[i]
                    stats_str = f" (Sim Mean: {s.get('mean', 0):.4f})"
                logger.debug(f"  View {i+1}: {p[:100]}...{stats_str}")

    def save_candidate_pool(self, path):
        """Save the entire raw candidate pool for observation."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.user_candidate_pools_observed, f, ensure_ascii=False, indent=2)
            logger.info(f"[Observation] Full candidate pool saved to {path} (Total users: {len(self.user_candidate_pools_observed)})")
        except Exception as e:
            logger.error(f"Error saving candidate pool: {e}")

    def save_views(self, path):
        """Save active views (prompts) and stats to a JSON file (User ID -> Views mapping)."""
        try:
            # Combine views and stats for saving
            data_to_save = {}
            for user_id, views in self.user_active_prompts.items():
                stats = getattr(self, "user_view_stats", {}).get(user_id, [])
                
                # If we have stats, save structure as list of dicts: [{"view": "...", "stats": {...}}]
                # If no stats (or empty), save as list of strings (backward compatible if needed, 
                # but better to normalize to dicts if we want to be consistent)
                
                # Let's standardize on list of dicts if any stats exist, or list of strings if legacy.
                # Actually, user requested saving stats. So let's save as structured objects.
                
                user_data = []
                for i, view in enumerate(views):
                    view_obj = {"view": view}
                    if stats and i < len(stats) and stats[i]:
                        view_obj["stats"] = stats[i]
                    user_data.append(view_obj)
                
                data_to_save[user_id] = user_data

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            if self.debug:
                logger.debug(f"Saved views and stats for {len(self.user_active_prompts)} users to {path}")
        except Exception as e:
            logger.error(f"Error saving views: {e}")

    def load_views(self, path):
        """Load active views from a JSON file and re-compute embeddings."""
        if not os.path.exists(path):
            logger.warning(f"Views file not found: {path}")
            return
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.user_active_prompts = {}
            if not hasattr(self, "user_view_stats"):
                self.user_view_stats = {}
            
            # Check format: Dict (new) or List (old)
            if isinstance(data, list):
                logger.warning("Loaded views are in old format (list). Ignoring.")
                pass
            elif isinstance(data, dict):
                # Handle both simple list of strings and new list of dicts
                for uid, user_data in data.items():
                    views = []
                    stats = []
                    if isinstance(user_data, list):
                        for item in user_data:
                            if isinstance(item, str):
                                views.append(item)
                                stats.append({})
                            elif isinstance(item, dict) and "view" in item:
                                views.append(item["view"])
                                stats.append(item.get("stats", {}))
                    
                    self.user_active_prompts[uid] = views
                    self.user_view_stats[uid] = stats
            
            if self.debug:
                logger.debug(f"Loaded views for {len(self.user_active_prompts)} users from {path}")
                
            # Re-compute embeddings for routing per user
            if self.schema_discovery.embedder:
                for uid, views in self.user_active_prompts.items():
                    if views:
                        self.user_active_prompt_embeddings[uid] = self.schema_discovery.embedder.encode(views)
            
            self.is_frozen = True # Assume if we load views, we are in frozen/inference mode
        except Exception as e:
            logger.error(f"Error loading views: {e}")

    def load_views_from_qdrant(self, save_path=None):
        """Recover active views from Qdrant metadata."""
        logger.info(f"Attempting to recover views from Qdrant collection: {self.collection_name}...")
        try:
            # We need to scroll through points to find unique 'view' fields in metadata
            # This is not efficient for huge datasets, but fine for recovery.
            # Qdrant scroll API
            client = self.mem0.vector_store.client
            collection = self.collection_name
            
            unique_views = set()
            next_offset = None
            
            # Limit scroll to avoid infinite loops if huge, say 5000 points check
            points_checked = 0
            limit = 10000
            
            while points_checked < limit:
                # scroll returns (points, next_offset)
                points, next_offset = client.scroll(
                    collection_name=collection,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                    
                for p in points:
                    payload = p.payload
                    if payload and "view" in payload:
                        # Try to find user_id in payload. mem0 usually stores it.
                        # If not present, maybe assign to 'unknown' or skip
                        uid = payload.get("user_id", "unknown")
                        unique_views.add((uid, payload["view"]))
                
                points_checked += len(points)
                if next_offset is None:
                    break
            
            # Reconstruct user_active_prompts
            self.user_active_prompts = defaultdict(list)
            for uid, view in unique_views:
                self.user_active_prompts[uid].append(view)
                
            logger.info(f"Recovered views for {len(self.user_active_prompts)} users from Qdrant.")
            
            # Save recovered views if path provided
            if save_path and self.user_active_prompts:
                self.save_views(save_path)

            # Re-compute embeddings
            if self.schema_discovery.embedder:
                for uid, views in self.user_active_prompts.items():
                     if views:
                         self.user_active_prompt_embeddings[uid] = self.schema_discovery.embedder.encode(views)
            
            self.is_frozen = True
                
        except Exception as e:
                logger.error(f"Error recovering views from Qdrant: {e}")

    def extract_batch(self, messages, user_id, metadata=None):
        """
        Extract information from messages using active views for the user.
        Returns a list of items to be stored.
        """
        # Prepare context text
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        items = []
        views = self.user_active_prompts.get(user_id, [])
        
        if not views:
            return []
            
        # Parallel execution for each view extraction
        # This is critical for vLLM performance as it allows batching on the server side
        def process_view(view):
            extraction = self._extract_with_llm(text, view)
            if extraction:
                return {
                    "extraction": extraction,
                    "user_id": user_id,
                    "metadata": metadata,
                    "view": view
                }
            return None

        # Use a reasonable max_workers to avoid thread explosion
        with ThreadPoolExecutor(max_workers=len(views)) as executor:
            futures = [executor.submit(process_view, view) for view in views]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        items.append(res)
                except Exception as e:
                    logger.error(f"Error in parallel extract_batch: {e}")
        
        return items

    def store_item(self, item):
        """
        Store a single extracted item into the vector store.
        """
        extraction = item.get("extraction")
        user_id = item.get("user_id")
        base_metadata = item.get("metadata") or {}
        view = item.get("view")
        
        if not extraction or not extraction.strip():
            return

        try:
            cleaned_extraction = self._clean_llm_json(extraction)
            try:
                extraction_json = json.loads(cleaned_extraction, strict=False)
            except json.JSONDecodeError:
                self.json_failure_count += 1
                logger.warning(f"JSON Parse Failure Count: {self.json_failure_count} (in store_item)")
                raise
            facts = extraction_json.get("facts", [])
            
            for fact_item in facts:
                if isinstance(fact_item, str):
                    try:
                        fact_dict = json.loads(fact_item, strict=False)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(fact_item, dict):
                    fact_dict = fact_item
                else:
                    continue

                content = fact_dict.pop("content", "")
                if not content:
                    continue
                    
                meta = base_metadata.copy()
                meta["view"] = view
                meta["type"] = "adaptive_view"
                meta["layer"] = "Adaptive"
                meta["role"] = "user"
                
                meta.update(fact_dict)
                
                # Direct add (Skip Consolidation logic as per pipeline design)
                # Retry logic for WinError 10048 (Port exhaustion)
                max_retries = 5
                base_delay = 1.0
                for attempt in range(max_retries):
                    try:
                        self.mem0.add(content, user_id=user_id, metadata=meta, infer=False)
                        break
                    except Exception as e:
                        # Check for socket/port exhaustion errors
                        error_str = str(e)
                        if "WinError 10048" in error_str or "Address already in use" in error_str:
                            if attempt < max_retries - 1:
                                sleep_time = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                                if self.debug:
                                    logger.warning(f"Port exhaustion (WinError 10048). Retrying in {sleep_time:.2f}s...")
                                time.sleep(sleep_time)
                                continue
                        # If it's another error or we ran out of retries, re-raise to be caught by outer block
                        raise e
                
        except Exception as e:
            logger.error(f"Error storing item: {e}")

    def add_memory(self, messages, user_id, metadata=None, skip_consolidation=False):
        """
        Phase 3: Extraction - Parallel Multi-View Extraction.
        Only runs when layer is frozen.
        """
        if not self.is_frozen:
            # Skip if not frozen (strict 2-phase)
            return
            
        self._multi_view_extract(messages, user_id, metadata, skip_consolidation=skip_consolidation)

    def extract_view(self, text, view_prompt):
        """
        Public wrapper to extract information from text using a specific view prompt.
        Useful for second-round extraction or manual extraction.
        """
        return self._extract_with_llm(text, view_prompt)

    def _clean_llm_json(self, text):
        """Clean LLM output to extract JSON from potential markdown blocks."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _process_single_view(self, prompt, text, user_id, metadata, skip_consolidation=False):
        """Helper for parallel execution of a single view extraction"""
        extraction = self._extract_with_llm(text, prompt)
        if extraction and extraction.strip():
            try:
                cleaned_extraction = self._clean_llm_json(extraction)
                extraction_json = json.loads(cleaned_extraction, strict=False)
                facts = extraction_json.get("facts", [])
                
                if not facts:
                    # Empty facts list, skip
                    # logger.debug(f"DEBUG: Empty facts for view '{prompt[:20]}...'")
                    return

                # Add each fact individually
                for fact_item in facts:
                    # fact_item could be a string (JSON representation) or a dict depending on LLM output consistency
                    if isinstance(fact_item, str):
                        try:
                            fact_dict = json.loads(fact_item, strict=False)
                        except json.JSONDecodeError:
                            if self.debug:
                                logger.debug(f"Inner JSON decode error: {fact_item[:50]}...")
                            continue
                    elif isinstance(fact_item, dict):
                        fact_dict = fact_item
                    else:
                        continue

                    # Extract content to be the memory text
                    content = fact_dict.pop("content", "")
                    if not content:
                        continue
                        
                    meta = metadata.copy() if metadata else {}
                    meta["view"] = prompt
                    meta["type"] = "adaptive_view"
                    meta["layer"] = "Adaptive"
                    meta["role"] = "user" # Ensure role is user
                    
                    # Merge other fact fields (like 'type', 'supporting_spans') into metadata
                    meta.update(fact_dict)
                    
                    if skip_consolidation:
                        # Direct add without search/update cycle
                        self.mem0.add(content, user_id=user_id, metadata=meta, infer=False)
                    else:
                        # Consolidate memory within the specific view
                        self._consolidate_memory_in_view(content, prompt, user_id, meta)
                    
            except json.JSONDecodeError:
                self.json_failure_count += 1
                logger.warning(f"JSON Parse Failure Count: {self.json_failure_count} (in _process_single_view)")
                if self.debug:
                    logger.debug(f"JSON decode error for view '{prompt[:20]}...': {extraction[:50]}...")
                pass

    def _consolidate_memory_in_view(self, content, view_name, user_id, metadata):
        """
        Simulate mem0's internal update logic but scoped to a specific view.
        Decides whether to ADD, UPDATE, or DELETE based on existing memories in the view.
        """
        if self.debug:
            logger.debug(f"Consolidation Enabled for view '{view_name}'")

        # 1. Search for existing memories within this view
        filters = {"view": view_name}
        try:
            search_results = self.mem0.search(
                query=content,
                user_id=user_id,
                filters=filters,
                limit=5
            )
        except Exception as e:
            error_str = str(e)
            if "Not found" in error_str or "doesn't exist" in error_str:
                search_results = {}
            elif "Wrong input" in error_str or "Conversion between multi and regular vectors failed" in error_str:
                logger.warning(f"Vector type mismatch in view '{view_name}'. Skipping search.")
                search_results = {}
            else:
                logger.error(f"Error searching memories for view '{view_name}': {e}")
                return

        existing_memories = search_results.get("results", [])
        
        if self.debug:
            logger.debug(f"Consolidate Search Query: '{content}'")
            logger.debug(f"Found {len(existing_memories)} existing memories for view '{view_name}'")
            for m in existing_memories:
                logger.debug(f"  - {m['memory']} (Score: {m.get('score', 'N/A')})")

        # Format existing memories for LLM
        # Use simple numeric IDs for LLM stability, map back to real IDs later
        existing_memories_text = ""
        id_map = {} # index -> real_id
        
        if existing_memories:
            for idx, mem in enumerate(existing_memories):
                id_map[idx] = mem['id']
                existing_memories_text += f"- ID: {idx}\n  Content: {mem['memory']}\n"
        else:
            existing_memories_text = "None"

        # 2. LLM Decision
        prompt = f"""
        You are a memory manager responsible for maintaining a consistent and concise memory state for a specific perspective.
        
        Perspective: {view_name}
        
        Your task is to evaluate a "New Memory" against "Existing Memories" and decide the appropriate action to maintain a clean knowledge base.
        
        Rules:
        - ADD: The new memory contains valuable new information NOT present in existing memories.
        - UPDATE: The new memory updates, corrects, or significantly refines an existing memory. Use this if the new memory conflicts with an existing one.
        - DELETE: The new memory explicitly requests deletion of an existing memory OR clearly supersedes an old state that is no longer valid.
        - NONE: The new memory is already fully covered by existing memories, is semantically identical, or is a subset of an existing memory. Do NOT add redundant information.
        
        Input:
        New Memory: {content}
        Existing Memories:
        {existing_memories_text}
        
        Output JSON format:
        {{
          "action": "ADD" | "UPDATE" | "DELETE" | "NONE",
          "memory_id": "target_memory_id_if_update_or_delete (use the numeric ID from the list, e.g., 0, 1, 2... or null for ADD)",
          "content": "final_memory_content_if_add_or_update"
        }}
        """

        try:
            # Construct guided_json schema for consolidation
            consolidation_schema = {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["ADD", "UPDATE", "DELETE", "NONE"]
                    },
                    "memory_id": {
                        "type": ["integer", "string", "null"]
                    },
                    "content": {
                        "type": "string"
                    }
                },
                "required": ["action"]
            }

            llm_kwargs = self._build_llm_create_kwargs(
                json_schema=consolidation_schema,
                schema_name="memory_consolidation",
            )
            response = call_llm_with_retry(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for memory management. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                #temperature=LLM_TEMPERATURE,
                # max_tokens=LLM_MAX_TOKENS,
                **llm_kwargs
            )
            result_text = response.choices[0].message.content.strip()

            # Defensive programming: Clean up markdown code blocks
            result_text = self._clean_llm_json(result_text)

            try:
                action_data = json.loads(result_text)
            except json.JSONDecodeError:
                self.json_failure_count += 1
                logger.warning(f"JSON Parse Failure Count: {self.json_failure_count} (in _consolidate_memory_in_view)")
                raise

            action = action_data.get("action", "ADD").upper()
            target_idx = action_data.get("memory_id")
            new_content = action_data.get("content", content)
            
            # Map numeric ID back to real ID
            target_id = None
            if target_idx is not None:
                # Handle string "0" or int 0
                try:
                    idx_val = int(target_idx)
                    target_id = id_map.get(idx_val)
                except (ValueError, TypeError):
                    if self.debug:
                        logger.debug(f"Invalid memory_id returned by LLM: {target_idx}")
            
            if self.debug:
                logger.debug(f"Consolidation Action for view '{view_name}': {action} -> {new_content[:50]}...")

            # Pre-check for exact duplicates to save calls and prevent redundancy
            if action == "ADD":
                for m in existing_memories:
                    if m['memory'].strip() == new_content.strip():
                        if self.debug:
                            logger.debug(f"Exact duplicate detected. Forcing NONE action.")
                        action = "NONE"
                        break

            # 3. Execute Action
            if action == "ADD":
                # Ensure we don't add duplicates if LLM made a mistake and said ADD for existing
                self.mem0.add(new_content, user_id=user_id, metadata=metadata, infer=False)
                
            elif action == "UPDATE" and target_id:
                # Update logic
                # Use internal _update_memory to ensure metadata (like 'view') is preserved/updated
                # mem0.update() public API does not support passing metadata and might lose custom fields
                try:
                    self.mem0._update_memory(target_id, new_content, existing_embeddings={}, metadata=metadata)
                except Exception as e:
                    logger.error(f"Error updating memory {target_id}: {e}")
                    # Fallback to delete + add if update fails
                    self.mem0.delete(target_id)
                    self.mem0.add(new_content, user_id=user_id, metadata=metadata, infer=False)
                
            elif action == "DELETE" and target_id:
                # Delete logic
                self.mem0.delete(target_id)
                
            elif action == "NONE":
                pass
                
        except Exception as e:
            logger.error(f"Error in memory consolidation for view '{view_name}': {e}")
            # Fallback to ADD if consolidation fails
            self.mem0.add(content, user_id=user_id, metadata=metadata, infer=False)


    def _multi_view_extract(self, messages, user_id, metadata=None, skip_consolidation=False):
        # Prepare context text
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Get active prompts for this user
        prompts = self.user_active_prompts.get(user_id, [])
        
        if not prompts:
            if self.debug:
                logger.debug(f"No active views found for user {user_id}. Skipping extraction.")
            return

        # Parallel execution for each prompt (View)
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [
                executor.submit(self._process_single_view, prompt, text, user_id, metadata, skip_consolidation)
                for prompt in prompts
            ]
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel view extraction: {e}")

    def _extract_with_llm(self, text, prompt):
        """
        Use LLM to extract information based on the specific view prompt.
        """
        extraction_prompt = f"""
        Perspective: {prompt}
        
        Extract stable facts and schemas from the following conversation matching the perspective.
        
        Each schema should include:
        - type: The category of information (e.g., persona, preference, fact, relationship, etc. relevant to the perspective)
        - content: The extracted information
        - supporting_spans: List of original message IDs or text snippets that support this schema
        
        Output Format:
        Return a JSON object with a single key "facts". 
        The value of "facts" must be a list of STRINGS, where each string is a valid JSON representation of the schema.

        Example format:
        {{
            "facts": [
                "{{\\"type\\": \\"preference\\", \\"content\\": \\"Alice prefers sci-fi movies\\", \\"supporting_spans\\": [\\"msg_123\\"]}}"
            ]
        }}
        
        Conversation:
        {text}
        
        Extract the relevant information concisely. If no relevant information is found, return nothing.
        """
        
        try:
            # Construct the guided_json schema
            guided_json_schema = {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "string" 
                        }
                    }
                },
                "required": ["facts"]
            }

            llm_kwargs = self._build_llm_create_kwargs(
                json_schema=guided_json_schema,
                schema_name="memory_extraction_facts",
            )
            response = call_llm_with_retry(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": extraction_prompt}],
                # temperature=LLM_TEMPERATURE,
                #max_tokens=LLM_MAX_TOKENS,
                **llm_kwargs
            )
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            if content.lower() in ["none", "nothing", "no information", ""]:
                return None
            return content
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return None
    
    def search(self, query, user_id, limit=100, top_n_views=1, limit_per_view=None, enable_view_routing=False):
        """
        Search interface for this layer with Dynamic Routing.
        """
        # Get user-specific embeddings
        active_embeddings = self.user_active_prompt_embeddings.get(user_id)
        active_prompts = self.user_active_prompts.get(user_id)
        
        # Dynamic Routing Phase
        if enable_view_routing and self.is_frozen and active_embeddings is not None and len(active_embeddings) > 0 and hasattr(self.schema_discovery, 'embedder'):
             try:
                 query_emb = self.schema_discovery.embedder.encode(query)
                 # Compute cosine similarity
                 # query_emb: (dim,), active_prompt_embeddings: (k, dim)
                 norm_prompts = np.linalg.norm(active_embeddings, axis=1)
                 norm_query = np.linalg.norm(query_emb)
                 
                 if norm_query > 0 and np.all(norm_prompts > 0):
                     sims = np.dot(active_embeddings, query_emb) / (norm_prompts * norm_query)
                     
                     # Get top N indices
                     # np.argsort returns ascending, so take last N and reverse
                     top_indices = np.argsort(sims)[-top_n_views:][::-1]
                     
                     all_results = []
                     seen_ids = set()
                     
                     # Search for each view and merge
                     view_stats = {}
                     
                     # Determine limit per view
                     view_limit = limit_per_view if limit_per_view is not None else limit

                     for idx in top_indices:
                         view = active_prompts[idx]
                         sim_score = float(sims[idx])
                         
                         # Attempt to filter by view
                         results = self.mem0.search(query, user_id=user_id, limit=view_limit, filters={"view": view})
                         if isinstance(results, dict) and "results" in results:
                             results = results["results"]
                         
                         view_stats[view] = {
                             "similarity": sim_score,
                             "count": len(results)
                         }
                             
                         for r in results:
                             # Try to find a unique ID
                             rid = r.get('id')
                             # Inject view into metadata if not present (or keep track of it)
                             # We want to record which view this memory came from
                             if "metadata" not in r:
                                 r["metadata"] = {}
                             r["metadata"]["_source_view"] = view
                             
                             # Fallback to content hash if no ID? 
                             # Ideally mem0 returns ID. If not, we might duplicate if same memory in multiple views (unlikely given ingestion logic but possible)
                             if rid:
                                 if rid not in seen_ids:
                                     seen_ids.add(rid)
                                     all_results.append(r)
                             else:
                                 # No ID, just append (risk of dupes)
                                 all_results.append(r)
                     
                     # Sort merged results by score descending
                     all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                     final_results = all_results[:limit]
                     
                     return {
                         "results": final_results,
                         "routing_info": {
                             "query": query,
                             "views": view_stats
                         }
                     }
             except Exception as e:
                 # Fallback to global search if routing/filtering fails
                 logger.error(f"Routing failed: {e}")
                 pass

        # Global Search
        results = self.mem0.search(query, user_id=user_id, limit=limit)
        
        # Normalize mem0 output (it might return a dict {'results': [...]})
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
            
        return {
            "results": results,
            "routing_info": {
                "query": query,
                "strategy": "global_fallback"
            }
        }
