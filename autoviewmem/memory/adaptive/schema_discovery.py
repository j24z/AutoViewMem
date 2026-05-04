import os
import json
import numpy as np
from typing import List, Dict
from sklearn.cluster import AgglomerativeClustering
import ast

from openai import OpenAI
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, EMBEDDING_NAME, EMBEDDING_BASE_URL, EMBEDDING_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..log_config import logger

from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import call_llm_with_retry

# Singleton storage for the embedder to avoid reloading model
_SHARED_EMBEDDER = None

class APIEmbedder:
    def __init__(self, base_url, api_key, model):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            embeddings = [d.embedding for d in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.array([])

class SchemaDiscovery:
    def __init__(self, llm_param_format="vllm", enable_thinking=False):
        """
        Initialize SchemaDiscovery with embedding model and LLM client.
        """
        global _SHARED_EMBEDDER
        
        if _SHARED_EMBEDDER is None:
            logger.info(f"Loading embedding model: {EMBEDDING_NAME}...")
            _SHARED_EMBEDDER = APIEmbedder(EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_NAME)
        
        self.embedder = _SHARED_EMBEDDER
        
        self.llm_client = OpenAI(
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL
        )
        if llm_param_format == "vllm":
            # For vLLM, we might want to use a different client or configuration if needed,
            # but currently ARK_BASE_URL points to the vLLM server.
            # We ensure the client is initialized correctly for the format.
            pass
            
        self.model = ARK_MODEL
        self.llm_param_format = llm_param_format
        self.enable_thinking = enable_thinking
        self.json_failure_count = 0

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


    def generate_candidate_prompts(self, text_chunk: str, prior_views: List[str] = None) -> List[str]:
        """
        Use Meta-Prompt to generate extraction instructions from text.
        """
        context_instruction = ""
        if prior_views:
            # Ensure we don't overwhelm the context if there are too many views
            # Take top 20 if too many
            display_views = prior_views[:20]
            views_str = "\n".join([f"- {v}" for v in display_views])
            context_instruction = f"""
        We have already analyzed the data using the following perspectives (Level 1 Views):
        {views_str}

        Please generate NEW, BROADER extraction instructions (Level 2 Views) that abstract over these details or find cross-cutting themes missed by the specific views.
        Focus on high-level patterns, summaries, and meta-analysis.
            """

        output_format_instruction = """
        Output Format:
        A JSON list of strings only.
        Return ONLY the JSON list. Do not include explanations.
        """
        if self.llm_param_format == "openai":
            output_format_instruction = """
        Output Format:
        A JSON object with a single key "prompts".
        The value of "prompts" must be a JSON list of strings.
        Return ONLY the JSON object. Do not include explanations.
        """

        meta_prompt = f"""
        You are an expert in information extraction for a comprehensive long-term memory system.
        Your goal is to extract ALL valuable information from the conversation, covering different granularities and dimensions.
        {context_instruction}
        Read the conversation snippet below. Generate 5-10 specific extraction instructions.
        The instructions should cover multiple layers of granularity:
        1. **Fact-level details**: Specific entities, dates, events, numbers, names (e.g., 'Extract the user's specific dietary restrictions mentioned').
        2. **Concept-level attributes**: User traits, preferences, habits, style (e.g., 'Extract the user's general communication style').
        3. **Relational information**: Connections between people, projects, or concepts.
        4. **Temporal/Process information**: Current state of tasks, plans, or timelines.
        5. **Implicit patterns**: Underlying intents, emotional states, or recurring themes.

        Each instruction must be an imperative sentence starting with 'Extract...'.
        Ensure the instructions are diverse and cover both fine-grained details and high-level summaries.

        Conversation Snippet:
        {text_chunk}

        {output_format_instruction}
        """

        # meta_prompt = f"""
        # Read the following conversation snippet. To build a long-term memory system capable of answering various questions about the user's preferences, relationships, event details, etc., what specific structured information do we need to extract from this dialogue?
        # Please list 5-10 different extraction instructions (Prompts). Each instruction should represent a unique perspective (e.g., extract timeline, analyze emotional shifts, extract implicit intent).
        
        # Conversation Snippet:
        # {text_chunk}
        
        # Output Format: JSON list of strings, e.g., ["Extract all specific time points and corresponding events", "Analyze the user's attitude change towards Speaker B", ...]
        # Return ONLY the JSON list, no other explanation. Ensure the 5-10 instructions are distinct and cover different dimensions (such as facts, emotions, relationships, intents, future plans, etc.). Please make sure to generate concrete instructions targeting specific aspects.
        # """

        try:
            if self.llm_param_format == "openai":
                list_schema = {
                    "type": "object",
                    "properties": {
                        "prompts": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["prompts"],
                }
            else:
                list_schema = {"type": "array", "items": {"type": "string"}}
            llm_kwargs = self._build_llm_create_kwargs(
                json_schema=list_schema,
                schema_name="candidate_prompts",
            )
            # Ensure temperature and max_tokens are set from config if not present
            if "temperature" not in llm_kwargs:
                llm_kwargs["temperature"] = LLM_TEMPERATURE
            if "max_tokens" not in llm_kwargs:
                llm_kwargs["max_tokens"] = LLM_MAX_TOKENS

            response = call_llm_with_retry(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": meta_prompt}],
                **llm_kwargs
            )
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                prompts = json.loads(content)
            except json.JSONDecodeError:
                self.json_failure_count += 1
                logger.warning(f"JSON Parse Failure Count: {self.json_failure_count} (in generate_candidate_prompts)")
                bracket_start = content.find("[")
                bracket_end = content.rfind("]")
                if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                    bracketed = content[bracket_start : bracket_end + 1].strip()
                    try:
                        prompts = json.loads(bracketed)
                    except json.JSONDecodeError:
                        try:
                            prompts = ast.literal_eval(bracketed)
                        except Exception:
                            prompts = None
                else:
                    prompts = None
                
                if not isinstance(prompts, list):
                    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    extracted = []
                    for ln in lines:
                        ln = ln.lstrip("-*• \t")
                        if ln and ln[0].isdigit() and "." in ln[:4]:
                            ln = ln.split(".", 1)[1].strip()
                        if not ln:
                            continue
                        if ln.lower().startswith("extract"):
                            extracted.append(ln)
                    prompts = extracted
            if isinstance(prompts, dict) and isinstance(prompts.get("prompts"), list):
                prompts = prompts["prompts"]

            if isinstance(prompts, list):
                # Ensure all are strings
                return [str(p) for p in prompts]
            return []
        except Exception as e:
            logger.error(f"Error generating candidate prompts: {e}")
            return []

    def _summarize_cluster_prompts(self, prompts: List[str]) -> str:
        """
        Summarize a list of similar prompts into a single representative prompt using LLM.
        """
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
            
        prompt_text = "\n".join([f"- {p}" for p in prompts])
        
        system_prompt = "You are an expert in summarizing memory extraction views."
        user_prompt = f"""
Here is a list of semantically similar memory extraction instructions (views) from a clustering algorithm:

{prompt_text}

Please summarize these into a single, comprehensive, and representative extraction instruction (view). 
The summary should cover the common core intent of these views while being concise and actionable.
Return ONLY the summarized instruction text.
"""
        try:
            llm_kwargs = self._build_llm_create_kwargs()
            # Ensure temperature and max_tokens are set from config if not present
            if "temperature" not in llm_kwargs:
                llm_kwargs["temperature"] = LLM_TEMPERATURE
            if "max_tokens" not in llm_kwargs:
                llm_kwargs["max_tokens"] = LLM_MAX_TOKENS
                
            response = call_llm_with_retry(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **llm_kwargs
            )
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content
        except Exception as e:
            logger.error(f"Error summarizing prompts: {e}")
            # Fallback to the first one or centroid logic (but here we just return first)
            return prompts[0]

    def select_orthogonal_prompts(self, candidate_prompts: List[str], k: int = 3) -> List[str]:
        """
        Select K orthogonal prompts using clustering and Global Library.
        """
        # Deduplicate first
        candidate_prompts = list(set(candidate_prompts))

        if not candidate_prompts:
            return []
            
        if len(candidate_prompts) <= k:
            selected_raw = candidate_prompts
        else:
            # Step A: Embed prompts
            embeddings = self.embedder.encode(candidate_prompts)
            
            # Step B: Clustering
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(embeddings)
            
            selected_raw = []
            
            # Helper to process a single cluster
            def process_cluster(label_idx):
                cluster_indices = np.where(labels == label_idx)[0]
                if len(cluster_indices) == 0:
                    return None
                
                # Use LLM to summarize the cluster
                cluster_prompts = [candidate_prompts[idx] for idx in cluster_indices]
                return self._summarize_cluster_prompts(cluster_prompts)

            # Step C: Summarize each cluster using LLM in parallel
            with ThreadPoolExecutor(max_workers=k) as executor:
                futures = {executor.submit(process_cluster, i): i for i in range(k)}
                
                # We want to maintain order if possible, or just collect all valid ones
                results = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        summary = future.result()
                        if summary:
                            results[idx] = summary
                    except Exception as e:
                        logger.error(f"Error processing cluster {idx}: {e}")
            
            # Collect results in order of cluster index
            for i in range(k):
                if i in results:
                    selected_raw.append(results[i])

        return selected_raw
