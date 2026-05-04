#!/usr/bin/env python3
"""
Adaptive Experiment Script: Train AdaptiveMemoryLayer using locomo_train dataset
"""
import os
import json
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
from tqdm import tqdm
import numpy as np
import networkx as nx
import random# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from autoviewmem.config import DATA_TRAIN_PATH, DATA_PATH
from autoviewmem.memory.adaptive.adaptive_ingestion import AdaptiveMemoryLayer
from autoviewmem.memory.log_config import logger

class AblationAdaptiveMemoryLayer(AdaptiveMemoryLayer):
    def __init__(self, ablation_mode="none", **kwargs):
        self.ablation_mode = ablation_mode
        if ablation_mode == "single_view":
            kwargs["k_views"] = 1
            logger.info("Ablation Mode: Single-view structured (k_views=1)")
        elif ablation_mode == "non_orthogonal":
            logger.info("Ablation Mode: Multi-view non-orthogonal (Similarity-based selection)")
        elif ablation_mode == "random":
            logger.info("Ablation Mode: w/o DPP (Random selection)")
        
        super().__init__(**kwargs)

    def _consolidate_single_user(self, user_id):
        # Default behavior for standard or single-view
        if self.ablation_mode == "none" or self.ablation_mode == "single_view":
            return super()._consolidate_single_user(user_id)

        # Logic for random and non_orthogonal
        if not self.user_candidate_pools[user_id]:
            if self.user_buffers[user_id]:
                self._generate_candidates_from_buffer(user_id)
            if not self.user_candidate_pools[user_id]:
                if self.debug:
                    logger.debug(f"No candidates to consolidate for {user_id}.")
                return

        if self.debug:
            logger.debug(f"Consolidating {len(self.user_candidate_pools[user_id])} candidate prompts for {user_id} (Mode: {self.ablation_mode})...")

        candidates = self.user_candidate_pools[user_id]
        # Dedup strings first
        candidates = list(set(candidates))
        k = self.k_views
        
        selected = []
        
        if self.ablation_mode == "random":
             if len(candidates) <= k:
                 selected = candidates
             else:
                 selected = random.sample(candidates, k)
                 
        elif self.ablation_mode == "non_orthogonal":
             # Select similar views (closest to centroid)
             if len(candidates) <= k:
                 selected = candidates
             elif self.schema_discovery.embedder:
                 try:
                     embeddings = self.schema_discovery.embedder.encode(candidates)
                     # Normalize
                     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                     embeddings = embeddings / (norms + 1e-9)
                     
                     # Centroid
                     centroid = np.mean(embeddings, axis=0)
                     centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                     
                     # Cosine similarity to centroid
                     sims = np.dot(embeddings, centroid)
                     
                     # Top K (highest similarity)
                     top_indices = np.argsort(sims)[-k:]
                     # argsort sorts ascending, so last k are highest
                     selected = [candidates[i] for i in top_indices]
                     # Reverse to have highest first
                     selected.reverse()
                 except Exception as e:
                     logger.error(f"Error in non-orthogonal selection: {e}. Fallback to random.")
                     selected = random.sample(candidates, k)
             else:
                 logger.warning("No embedder available for non-orthogonal selection. Fallback to random.")
                 selected = random.sample(candidates, k)

        # --- Common Storage Logic ---
        final_views = selected
        final_stats = [{}] * len(selected)

        self.user_active_prompts[user_id] = final_views
        
        if not hasattr(self, "user_view_stats"):
            self.user_view_stats = {}
        self.user_view_stats[user_id] = final_stats
        
        # Cache embeddings
        if self.schema_discovery.embedder:
            try:
                self.user_active_prompt_embeddings[user_id] = self.schema_discovery.embedder.encode(final_views)
            except Exception as e:
                logger.error(f"Error encoding final views: {e}")
        
        self.user_candidate_pools[user_id] = []
        self.user_buffers[user_id] = []
        
        if self.debug:
            logger.debug(f"Adaptive Layer Frozen for {user_id} with {len(self.user_active_prompts[user_id])} views.")

class AdaptiveLayerExperiment:
    def __init__(self, data_path=None, warm_start_steps=50, k_views=10, debug=False, collection_name="autoviewmem_demo", enable_consolidation=True, force=False, use_dpp=False, llm_param_format="vllm", ablation_mode="none"):
        # Initialize only the AdaptiveMemoryLayer
        self.adaptive_layer = AblationAdaptiveMemoryLayer(ablation_mode=ablation_mode, warm_start_steps=warm_start_steps, k_views=k_views, debug=debug, collection_name=collection_name, use_dpp=use_dpp, llm_param_format=llm_param_format)
        
        self.data_path = data_path
        self.data = None
        self.debug = debug
        self.enable_consolidation = enable_consolidation
        self.force = force
        self.checkpoint_lock = threading.Lock()
        
        # Shared executors to prevent thread churn
        self.speaker_executor = ThreadPoolExecutor(max_workers=500)
        self.llm_executor = ThreadPoolExecutor(max_workers=500) # High concurrency for LLM
        self.db_executor = ThreadPoolExecutor(max_workers=500) # Limited concurrency for Local DB/Embedding

        if data_path:
            self.load_data()

    def load_data(self):
        """Load dataset"""
        logger.info(f"Loading data from {self.data_path}...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.data = self._normalize_data(raw_data)
        if self.debug:
            logger.debug(f"总共有 {len(self.data)} 轮会话")
        return self.data

    def _normalize_data(self, raw_data):
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            return raw_data
        sample = raw_data[0]
        if isinstance(sample, dict) and "conversation" in sample:
            return raw_data
            
        # Check for Locomo format (list of dicts with session_X keys)
        if isinstance(sample, dict) and any(k.startswith("session_") for k in sample.keys()):
            # Locomo format: list of dicts, each dict has "session_X" keys which are lists of dicts
            normalized = []
            for item in raw_data:
                conversation = {
                    "speaker_a": "Caroline", 
                    "speaker_b": "Melanie"
                }
                # Copy session fields
                for k, v in item.items():
                    if k.startswith("session_") and not k.endswith("_observation"):
                        conversation[k] = v
                    elif "date_time" in k:
                         conversation[k] = v
                
                normalized.append({
                    "conversation": conversation,
                    "qa": item.get("qa")
                })
            return normalized

        if isinstance(sample, dict) and "haystack_sessions" in sample:
            normalized = []
            for item in raw_data:
                sessions = item.get("haystack_sessions") or []
                dates = item.get("haystack_dates") or []
                conversation = {
                    "speaker_a": "user",
                    "speaker_b": "assistant"
                }
                for i, session in enumerate(sessions):
                    key = f"session_{i}"
                    chats = []
                    for msg in session:
                        role = msg.get("role")
                        if role not in ("user", "assistant"):
                            continue
                        chats.append({
                            "speaker": role,
                            "text": msg.get("content", "")
                        })
                    conversation[key] = chats
                    if i < len(dates):
                        conversation[f"{key}_date_time"] = dates[i]
                normalized.append({
                    "conversation": conversation,
                    "question_id": item.get("question_id"),
                    "question": item.get("question"),
                    "answer": item.get("answer")
                })
            return normalized
        return raw_data

    def calculate_total_batches(self):
        """Calculate total number of batches across all conversations"""
        total_batches = 0
        batch_size = 2  # Keep consistent with _ingest_stream
        
        if not self.data:
            return 0
            
        for item in self.data:
            conversation = item["conversation"]
            # Loop through sessions
            for key in conversation.keys():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                    continue
                
                chats = conversation[key]
                num_msgs = len(chats)
                
                # Each chat generates 1 message for speaker A and 1 for speaker B
                # So we have num_msgs for A and num_msgs for B
                
                # Batches for A
                batches_a = (num_msgs + batch_size - 1) // batch_size
                # Batches for B
                batches_b = (num_msgs + batch_size - 1) // batch_size
                
                total_batches += batches_a + batches_b
                
        return total_batches

    def _load_checkpoint(self, filename):
        """Load processed indices from checkpoint file"""
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
        processed = set()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            processed.add(int(line))
            except Exception as e:
                logger.error(f"Error loading checkpoint {filename}: {e}")
        return processed

    def _append_checkpoint(self, filename, idx):
        """Append processed index to checkpoint file"""
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
        try:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(f"{idx}\n")
        except Exception as e:
            logger.error(f"Error writing to checkpoint {filename}: {e}")

    def _calculate_conversation_batches(self, conversation):
        """Calculate batches for a single conversation"""
        batch_size = 2
        total = 0
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            chats = conversation[key]
            num_msgs = len(chats)
            batches_a = (num_msgs + batch_size - 1) // batch_size
            batches_b = (num_msgs + batch_size - 1) // batch_size
            total += batches_a + batches_b
        return total

    def process_conversation(self, item, idx, mode="extraction", position=0, pbar=None):
        """Process a single conversation item"""
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        # Unique User IDs for isolation
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # Iterate through sessions in the conversation
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            
            # Extract timestamp if available
            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[key]

            messages_a = []
            messages_b = []
            
            # Handle list of dicts (standard/Locomo format)
            if isinstance(chats, list):
                for chat in chats:
                    # Extract source ID if available (dia_id for Locomo)
                    source_id = chat.get("dia_id", chat.get("id", ""))
                    
                    if chat["speaker"] == speaker_a:
                        msg = {"role": "user", "content": f"{speaker_a}: {chat['text']}"}
                        if source_id: msg["_source_id"] = source_id
                        messages_a.append(msg)
                        
                        msg = {"role": "assistant", "content": f"{speaker_a}: {chat['text']}"}
                        if source_id: msg["_source_id"] = source_id
                        messages_b.append(msg)
                        
                    elif chat["speaker"] == speaker_b:
                        msg = {"role": "assistant", "content": f"{speaker_b}: {chat['text']}"}
                        if source_id: msg["_source_id"] = source_id
                        messages_a.append(msg)
                        
                        msg = {"role": "user", "content": f"{speaker_b}: {chat['text']}"}
                        if source_id: msg["_source_id"] = source_id
                        messages_b.append(msg)
                    else:
                        # Skip unknown speakers or handle error
                        continue
            
            # Handle dict of lists (observation/summary format) - legacy or specific use case
            elif isinstance(chats, dict):
                 pass # Currently skipping or implement if needed for observation fields

            # Process for Speaker A and Speaker B in parallel
            futures = []
            futures.append(self.speaker_executor.submit(self._ingest_stream, speaker_a_user_id, messages_a, timestamp, mode=mode, pbar=pbar))
            futures.append(self.speaker_executor.submit(self._ingest_stream, speaker_b_user_id, messages_b, timestamp, mode=mode, pbar=pbar))
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing speaker stream: {e}")

        # Update checkpoint if successful
        if mode == "discovery":
            # Incremental consolidation to save memory
            try:
                self.adaptive_layer.consolidate_discovery(user_id=speaker_a_user_id)
                self.adaptive_layer.consolidate_discovery(user_id=speaker_b_user_id)
            except Exception as e:
                logger.error(f"Error in incremental consolidation: {e}")

    def _ingest_stream(self, user_id, messages, timestamp, mode="extraction", pbar=None):
        """Helper to feed messages into Adaptive Layer"""
        # We feed messages one by one or in small batches to simulate a stream
        # AdaptiveLayer expects a list of messages (a turn or a batch)
        # Here we treat the whole session as a sequence of turns.
        
        # Strategy: Feed messages in pairs (User + Assistant) or individually?
        # AdaptiveLayer logic accumulates them in buffer anyway.
        # Let's feed them in chunks of 2 (User query + Assistant response roughly)
        batch_size = 2
        
        # Prepare batches
        batches = []
        for i in range(0, len(messages), batch_size):
            chunk = messages[i : i + batch_size]
            batches.append(chunk)

        # Process batches
        # Parallelism is handled at Conversation, Speaker, and View levels.
        
        def process_batch(chunk_data):
            # Extract source_ids
            source_ids = []
            clean_chunk = []
            for msg in chunk_data:
                msg_copy = msg.copy()
                if "_source_id" in msg_copy:
                    source_ids.append(msg_copy.pop("_source_id"))
                clean_chunk.append(msg_copy)
            
            metadata = {"timestamp": timestamp, "source_ids": source_ids}
            try:
                if mode == "discovery":
                    self.adaptive_layer.ingest_for_discovery(clean_chunk, user_id, metadata=metadata)
                else:
                    # If consolidation is disabled, we skip the search-update cycle
                    skip = not self.enable_consolidation
                    self.adaptive_layer.add_memory(clean_chunk, user_id, metadata=metadata, skip_consolidation=skip)
            except Exception as e:
                logger.error(f"Error in batch ingestion: {e}")
            
            if pbar:
                pbar.update(1)

        def process_batch_pipeline(chunk_data):
            """Pipeline: LLM Extraction -> DB Storage"""
            # Extract source_ids
            source_ids = []
            clean_chunk = []
            for msg in chunk_data:
                msg_copy = msg.copy()
                if "_source_id" in msg_copy:
                    source_ids.append(msg_copy.pop("_source_id"))
                clean_chunk.append(msg_copy)
            
            metadata = {"timestamp": timestamp, "source_ids": source_ids}
            try:
                # 1. LLM Extraction (Blocking but High Concurrency)
                items = self.adaptive_layer.extract_batch(clean_chunk, user_id, metadata=metadata)
                
                # 2. Submit to DB Executor (Fire and Forget or Wait?)
                # We iterate and submit each item to DB executor
                for item in items:
                    self.db_executor.submit(self.adaptive_layer.store_item, item)
                    
            except Exception as e:
                logger.error(f"Error in batch pipeline: {e}")
            
            if pbar:
                pbar.update(1)

        if not self.enable_consolidation and mode == "extraction":
            # Parallel mode: Process all batches in this stream concurrently using Pipeline
            futures = [self.llm_executor.submit(process_batch_pipeline, chunk) for chunk in batches]
            for future in futures:
                future.result() # Wait for LLM extraction to finish (Storage happens in background)
        else:
            # Serial mode: Process batches sequentially
            # Required for discovery (stateful buffer) and consolidation (search-update consistency)
            for chunk in batches:
                process_batch(chunk)

    def run_discovery_phase(self, max_workers=20):
        """Phase 1: Divergence (Discovery)"""
        # Ensure we are not frozen (in case views were loaded)
        self.adaptive_layer.is_frozen = False
        
        if not self.data:
            raise ValueError("No data loaded.")
        logger.info(f"Starting Phase 1: Discovery on {len(self.data)} conversations...")
        
        total_batches = self.calculate_total_batches()
        
        with tqdm(total=total_batches, desc="Discovery Phase (Batches)") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_conversation, item, idx, mode="discovery", pbar=pbar) for idx, item in enumerate(self.data)]
                for future in futures:
                     try:
                         future.result()
                     except Exception as e:
                         logger.error(f"Error processing conversation in discovery: {e}")

    def finalize_discovery(self):
        """Phase 2: Convergence (Consolidation)"""
        logger.info("Starting Phase 2: Convergence (Consolidation)...")
        self.adaptive_layer.flush_discovery_buffers()
        self.adaptive_layer.consolidate_discovery()
        
        # Save views
        views_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "adaptive/adaptive_views.json"))
        logger.info(f"Saving views to {views_path}...")
        self.adaptive_layer.save_views(views_path)

    def run_extraction_phase(self, max_workers=20):
        """Phase 3: Extraction"""
        if not self.data:
            raise ValueError("No data loaded.")
        logger.info(f"Starting Phase 3: Extraction on {len(self.data)} conversations...")
        
        checkpoint_filename = "extraction_checkpoint.jsonl"
        
        # Handle Force Mode
        if self.force:
            logger.info("Force mode enabled. Resetting checkpoint...")
            checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), checkpoint_filename))
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Deleted checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint: {e}")
        
        # Load checkpoint
        processed_indices = self._load_checkpoint(checkpoint_filename)
        logger.info(f"Found {len(processed_indices)} processed conversations in checkpoint. Resuming...")
        
        # Calculate remaining batches
        total_batches = 0
        tasks = []
        
        for idx, item in enumerate(self.data):
            if idx in processed_indices:
                continue
            
            # Calculate batches for this conversation to set pbar correctly
            batches = self._calculate_conversation_batches(item["conversation"])
            total_batches += batches
            tasks.append((idx, item))
            
        logger.info(f"Remaining tasks: {len(tasks)} conversations, approx {total_batches} batches.")
        
        with tqdm(total=total_batches, desc="Extraction Phase (Batches)") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_conversation, item, idx, mode="extraction", pbar=pbar) for idx, item in tasks]
                for future in futures:
                     try:
                         future.result()
                     except Exception as e:
                         logger.error(f"Error processing conversation in extraction: {e}")

    def _build_similarity_graph(self, vectors, threshold=0.8):
        """
        Build a graph where edges connect memories with cosine similarity > threshold.
        Returns a list of connected components (clusters of indices).
        """
        n = len(vectors)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        # Build Graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Find pairs with similarity > threshold
        rows, cols = np.where(np.triu(similarity_matrix, k=1) > threshold)
        
        for r, c in zip(rows, cols):
            G.add_edge(r, c)
            
        # Get connected components
        clusters = []
        for component in nx.connected_components(G):
            clusters.append(list(component))
            
        return clusters

    def _process_single_user_round_2(self, user_id, max_workers):
        """Process a single user for Round 2 (L2 Discovery & Extraction)"""
        # 1. Fetch L1 Memories with Vectors
        try:
            # We need vectors for clustering. mem0.get_all might not return them by default.
            # Access Qdrant client directly.
            client = self.adaptive_layer.mem0.vector_store.client
            collection_name = self.adaptive_layer.collection_name
            
            # Scroll all points for user
            points = []
            offset = None
            from qdrant_client.http import models
            
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            )
            
            while True:
                res = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset
                )
                batch_points, offset = res
                points.extend(batch_points)
                if offset is None:
                    break
            
            if not points:
                return None

            l2_max_points = int(os.environ.get("L2_MAX_POINTS_PER_USER", "3000"))
            l2_sim_graph_max_n = int(os.environ.get("L2_SIM_GRAPH_MAX_N", "400"))
            l2_chunk_size = int(os.environ.get("L2_CLUSTER_CHUNK_SIZE", "80"))
            l2_max_clusters = int(os.environ.get("L2_MAX_CLUSTERS_PER_USER", "120"))
            l2_max_tasks = int(os.environ.get("L2_MAX_TASKS_PER_USER", "600"))

            if len(points) > l2_max_points:
                sampled_idx = random.sample(range(len(points)), l2_max_points)
                points = [points[i] for i in sampled_idx]

            logger.info(f"[L2] user={user_id} points={len(points)}")
                
            # Extract Text and Vectors
            memory_texts = []
            vectors = []
            
            for p in points:
                # Payload data
                val = p.payload.get("data", "") or p.payload.get("memory", "")
                if isinstance(val, (dict, list)):
                     val = json.dumps(val, ensure_ascii=False)
                memory_texts.append(str(val))
                vec = p.vector
                if isinstance(vec, dict):
                    try:
                        vec = next(iter(vec.values()))
                    except StopIteration:
                        vec = None
                if vec is None:
                    continue
                vectors.append(vec)
                
            vectors = np.array(vectors)
            if vectors.ndim != 2 or len(vectors) == 0:
                return None
            
            # ---------------------------------------------------------
            # Step 1: View Discovery (Scanning with Density 60)
            # ---------------------------------------------------------
            l1_views = self.adaptive_layer.user_active_prompts.get(user_id, [])
            
            # Batch size 60
            batch_size = 60
            candidate_pool = []
            
            # Create batches
            batches = [memory_texts[i:i + batch_size] for i in range(0, len(memory_texts), batch_size)]
            
            def process_view_generation(batch_txt):
                context = "\n".join(batch_txt)
                # Generate candidates using L1 views as prior
                cands = self.adaptive_layer.schema_discovery.generate_candidate_prompts(
                    context, prior_views=l1_views
                )
                return cands
                
            # Run in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_view_generation, b) for b in batches]
                for future in as_completed(futures):
                    try:
                        cands = future.result()
                        if cands:
                            candidate_pool.extend(cands)
                    except Exception as e:
                        logger.error(f"Error generating views for batch: {e}")
                        
            # Deduplicate Candidates (DPP)
            # Ensure strings
            candidate_pool = [str(c) if isinstance(c, (dict, list)) else c for c in candidate_pool]
            
            if not candidate_pool:
                # logger.info(f"No L2 candidates generated for {user_id}.")
                return None
                
            if self.adaptive_layer.use_dpp:
                 l2_views = self.adaptive_layer.select_dpp_prompts(
                    candidate_pool, k=min(self.adaptive_layer.k_views, 10)
                )
            else:
                l2_views = self.adaptive_layer.schema_discovery.select_orthogonal_prompts(
                    candidate_pool, k=min(self.adaptive_layer.k_views, 10)
                )
            
            # Normalize l2_views (extract string if dict from DPP)
            final_l2_views = []
            for v in l2_views:
                if isinstance(v, dict) and "view" in v:
                    final_l2_views.append(v["view"])
                elif isinstance(v, str):
                    final_l2_views.append(v)
            l2_views = final_l2_views
            
            # ---------------------------------------------------------
            # Step 2: Content Extraction (Clustering)
            # ---------------------------------------------------------
            
            # Cluster memories
            if len(vectors) <= l2_sim_graph_max_n:
                clusters = self._build_similarity_graph(vectors, threshold=0.8)
            else:
                indices = list(range(len(vectors)))
                random.shuffle(indices)
                clusters = [indices[i:i + l2_chunk_size] for i in range(0, len(indices), l2_chunk_size)]

            if len(clusters) > l2_max_clusters:
                clusters = random.sample(clusters, l2_max_clusters)
            
            # For each cluster, extract summaries for each view
            user_summaries = {} # View -> List of summaries
            
            # We want to process each cluster against ALL views.
            # If we have C clusters and K views, we have C*K tasks.
            
            extraction_tasks = []
            
            for cluster_indices in clusters:
                # If cluster > 30, sample 30
                if len(cluster_indices) > 30:
                    selected_indices = random.sample(cluster_indices, 30)
                else:
                    selected_indices = cluster_indices
                    
                # Build context
                cluster_text = [memory_texts[i] for i in selected_indices]
                
                for view in l2_views:
                    extraction_tasks.append((view, cluster_text))
                    if len(extraction_tasks) >= l2_max_tasks:
                        break
                if len(extraction_tasks) >= l2_max_tasks:
                    break

            logger.info(f"[L2] user={user_id} views={len(l2_views)} clusters={len(clusters)} tasks={len(extraction_tasks)} workers={max_workers}")
                    
            def process_cluster_extraction(view, cluster_txt):
                context = "\n".join(cluster_txt)
                summary = self.adaptive_layer.extract_view(context, view)
                
                # Real-time Storage: Save L2 summary to Vector Store immediately
                if summary:
                    try:
                        # Clean and parse JSON
                        clean_str = self.adaptive_layer._clean_llm_json(summary)
                        if clean_str:
                            try:
                                summary_json = json.loads(clean_str)
                            except json.JSONDecodeError:
                                self.adaptive_layer.json_failure_count += 1
                                logger.warning(f"JSON Parse Failure Count: {self.adaptive_layer.json_failure_count} (in experiment L2 summary)")
                                raise

                            facts = summary_json.get("facts", [])
                            
                            for fact in facts:
                                if isinstance(fact, str):
                                    try:
                                        fact = json.loads(fact)
                                    except json.JSONDecodeError:
                                        self.adaptive_layer.json_failure_count += 1
                                        logger.warning(f"JSON Parse Failure Count: {self.adaptive_layer.json_failure_count} (in experiment L2 fact)")
                                        fact = {"content": fact}
                                    except Exception:
                                        fact = {"content": fact}
                                
                                content = fact.get("content")
                                if not content:
                                    continue
                                    
                                # Construct L2 Metadata
                                metadata = {
                                    "view": view,
                                    "layer": "Adaptive_L2", # Mark as Level 2
                                    "type": "summary",      # Mark as summary
                                    "source_level": "L1_aggregation",
                                    "role": "user"
                                }
                                # Merge other fact fields
                                metadata.update({k: v for k, v in fact.items() if k != "content"})
                                
                                # Add to memory (Directly, infer=False for speed since it is already extracted)
                                self.adaptive_layer.mem0.add(content, user_id=user_id, metadata=metadata, infer=False)
                    except Exception as e:
                        logger.error(f"Error saving L2 memory in real-time: {e}")
                            
                return view, summary

            # Run extraction in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for view, txt in extraction_tasks:
                    futures.append(executor.submit(process_cluster_extraction, view, txt))
                    
                for future in as_completed(futures):
                    try:
                        v, s = future.result()
                        if s:
                            if v not in user_summaries:
                                user_summaries[v] = []
                            user_summaries[v].append(s)
                    except Exception as e:
                        logger.error(f"Error in L2 extraction: {e}")
            
            return {
                "l2_views": l2_views,
                "summaries": user_summaries
            }
            
        except Exception as e:
            logger.error(f"Error processing {user_id} in Round 2: {e}")
            import traceback
            logger.exception("Exception in Round 2 processing")
            return None

    def run_second_round_processing(self, output_file="second_round_summary.json", max_workers=5):
        """
        Execute Second Round: L2 Discovery & Summarization based on L1 memories.
        """
        logger.info("\nStarting Second Round Processing (L2 Discovery & Summarization)...")
        
        # 1. Identify Users
        # We scan active prompts to find users who have been processed
        user_ids = list(self.adaptive_layer.user_active_prompts.keys())
        if not user_ids:
            logger.warning("No active users found in Adaptive Layer. Skipping Second Round.")
            return

        results = {}
        
        # Determine concurrency
        total_users = len(user_ids)
        concurrent_users = min(20, total_users) if total_users > 0 else 1
        inner_workers = max(2, max_workers // concurrent_users)
        
        logger.info(f"Processing {total_users} users with {concurrent_users} concurrent users and {inner_workers} inner threads each.")

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = {executor.submit(self._process_single_user_round_2, uid, inner_workers): uid for uid in user_ids}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Users (Round 2)"):
                user_id = futures[future]
                try:
                    res = future.result()
                    if res:
                        results[user_id] = res
                except Exception as e:
                    logger.error(f"Error in user processing future {user_id}: {e}")

        # 5. Output and Storage
        # Generate path with date
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        date_str = time.strftime("%Y%m%d")
        
        # Determine output directory
        results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"))
        daily_dir = os.path.join(results_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        
        # Generate filename
        base_name = os.path.splitext(output_file)[0]
        filename = f"{base_name}_{timestamp}.json"
        
        output_path = os.path.join(daily_dir, filename)
        
        logger.info(f"Saving Second Round results to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def shutdown(self):
        """Shutdown all executors"""
        logger.info("Shutting down executors...")
        self.speaker_executor.shutdown(wait=True)
        self.llm_executor.shutdown(wait=True)
        self.db_executor.shutdown(wait=True)
        logger.info("Executors shut down.")

def monitor_memory(interval=2):
    process = psutil.Process(os.getpid())
    logger.info(f"Starting Memory Monitor (PID: {os.getpid()})...")
    try:
        while True:
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            logger.info(f"[Memory Monitor] RSS: {rss_mb:.2f} MB")
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Memory monitor stopped: {e}")

def main():
    import argparse
    
    # Argument Parser setup
    parser = argparse.ArgumentParser(description="Run Adaptive Memory Experiment Phases")
    parser.add_argument("--mode", type=str, default="ingest", choices=["all", "ingest", "extract", "l2"], 
                        help="Execution mode: 'all' (Stages 1-4), 'ingest' (Stages 1-3: Discovery & Extraction), 'extract' (Stage 3: Extraction only), 'l2' (Stage 4: L2 Summarization)")
    parser.add_argument("--test", action="store_true", help="Run on a small subset of data for testing")
    parser.add_argument("--force", action="store_true", default=True, help="Force re-run (reset collection and views). Default is True.")
    parser.add_argument("--no_force", action="store_false", dest="force", help="Skip force re-run")
    parser.add_argument("--use_existing_views", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=False, help="Load adaptive_views.json and skip view regeneration when available (default: False). Accepts true/false")
    parser.add_argument("--no_dpp", action="store_false", dest="use_dpp", default=True, help="Disable DPP for view selection")
    parser.add_argument("--collection_name", type=str, default="automem_5_4_train_8B", help="Qdrant collection name (default: adaptive_memory_layer_1_29)")
    parser.add_argument("--no_consolidation", action="store_false", dest="enable_consolidation", default=False, 
                        help="Disable consolidation during extraction (default: True)")
    parser.add_argument("--llm_param_format", type=str, default="vllm", choices=["vllm", "openai"], help="LLM 参数格式：vllm=extra_body+开启思考；openai=response_format JSON 约束 + reasoning_effort=none")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "single_view", "non_orthogonal", "random"], 
                        help="Ablation mode: single_view (structured k=1), non_orthogonal (multi-view similar), random (w/o DPP)")
    args = parser.parse_args()

    # Determine which stages to run
    run_discovery = args.mode in ["all", "ingest"]
    run_extract = args.mode in ["all", "ingest", "extract"]
    run_l2 = args.mode in ["all", "l2"]

    logger.info(f"Execution Mode: {args.mode}")
    logger.info(f"Collection Name: {args.collection_name}")
    logger.info(f"Force Mode: {args.force}")
    logger.info(f"Use Existing Views: {args.use_existing_views}")
    logger.info(f"Use DPP: {args.use_dpp}")
    logger.info(f"Enable Consolidation: {args.enable_consolidation}")
    logger.info(f"  - Run Discovery (Phases 1-2): {run_discovery}")
    logger.info(f"  - Run Extraction (Phase 3): {run_extract}")
    logger.info(f"  - Run L2 Summarization (Phase 4): {run_l2}")

    # Start memory monitoring
    monitor_thread = threading.Thread(target=monitor_memory, args=(5,), daemon=True)
    monitor_thread.start()

    # Configuration
    # Load data if we are running discovery or extraction (Phases 1-3)
    should_load_data = run_discovery or run_extract
    data_path = "data/locomo/locomo_train.json" if should_load_data else None
    
    debug = bool(args.test)
    if args.test:
        debug = True
        logger.info("Test mode enabled: Will run on subset of data if loading data.")

    # Initialize Experiment
    # warm_start_steps=10: Lower threshold for testing to trigger discovery faster per user
    # enable_consolidation=False: Enable fast parallel extraction without deduplication
    # UPDATED: Use existing collection "adaptive_memory_layer_1_29" for Second Round
    experiment = AdaptiveLayerExperiment(
        data_path=data_path, 
        warm_start_steps=50, 
        k_views=10, 
        debug=debug, 
        collection_name=args.collection_name, 
        enable_consolidation=args.enable_consolidation,
        force=args.force,
        use_dpp=args.use_dpp,
        llm_param_format=args.llm_param_format,
        ablation_mode=args.ablation
    )
    
    # Reset collection to ensure clean state
    if args.force and (run_discovery or run_extract):
        experiment.adaptive_layer.reset()
    
    # Load existing views for Second Round or Resume
    views_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "adaptive/adaptive_views.json"))
    has_views = False
    
    if os.path.exists(views_path):
        if args.use_existing_views or not args.force:
            logger.info(f"Loading existing views from {views_path}...")
            experiment.adaptive_layer.load_views(views_path)
            has_views = True
        else:
            logger.info(f"Force mode: Ignoring existing views at {views_path}")
    else:
        if (run_extract or run_l2) and not run_discovery:
             logger.warning("Warning: Views file not found and Discovery skipped. Extraction/Stage 4 may fail or do nothing.")
        
        if run_discovery:
             logger.info("Views file not found. Will run Discovery Phase.")
    
    # Run Processing
    start_time = time.time()
    
    if run_discovery:
        # Apply test slicing if enabled
        if args.test and experiment.data:
            logger.info("Test mode: Slicing data to first 2 conversations.")
            experiment.data = experiment.data[:2] 
        
        # Phase 1 & 2: Discovery & Convergence
        # Only run if we don't have views or if forced
        if not has_views or (args.force and not args.use_existing_views):
            # Phase 1: Divergence
            experiment.run_discovery_phase(max_workers=100)
            
            # Save raw candidate pool for observation
            pool_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "adaptive/candidate_pool.json"))
            experiment.adaptive_layer.save_candidate_pool(pool_path)
            
            # Phase 2: Convergence
            experiment.finalize_discovery()
        else:
            logger.info("Views already loaded. Skipping Discovery and Consolidation phases.")

    if run_extract:
        # Phase 3: Extraction
        logger.info("\n[Phase 3] Starting Extraction...")
        experiment.run_extraction_phase(max_workers=50) 
    
    
    if run_l2:
        # Phase 4: Second Round (L2 Summarization)
        # Uses batch_size=50 and broader prompt based on L1 views
        # Increased max_workers to 50 for higher concurrency on LLM calls
        experiment.run_second_round_processing(max_workers=80)
    
    experiment.shutdown()

    end_time = time.time()
    
    logger.info("\n" + "="*50)
    logger.info("JSON Parse Failure Statistics:")
    logger.info(f"  - Schema Discovery Failures: {experiment.adaptive_layer.schema_discovery.json_failure_count}")
    logger.info(f"  - Adaptive Layer Failures:   {experiment.adaptive_layer.json_failure_count}")
    logger.info(f"  - Total Failures:            {experiment.adaptive_layer.schema_discovery.json_failure_count + experiment.adaptive_layer.json_failure_count}")
    logger.info("="*50 + "\n")

    logger.info(f"\nProcessing complete! Time taken: {end_time - start_time:.2f} seconds")
    

if __name__ == "__main__":
    main()
