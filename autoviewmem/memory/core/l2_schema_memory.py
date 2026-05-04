import os
import json
from mem0 import Memory
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, EMBEDDING_NAME, EMBEDDING_SIZE, EMBEDDING_BASE_URL, EMBEDDING_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..log_config import logger

class L2SchemaMemory:
    def __init__(self):
        logger.info("Initializing L2SchemaMemory...")
        llm_model = ARK_MODEL
        llm_base_url = ARK_BASE_URL
        llm_api_key = ARK_API_KEY

        self.schema_prompt = """
        Extract stable facts and schemas from the following conversation. Focus on:
        - Persona: Long-term personality traits, values, and characteristics
        - Preferences: Consistent likes and dislikes
        - Stable facts: Information that doesn't change over time
        - Relationships: Connections between people or entities
        
        Each schema should include:
        - type: persona, preference, fact, or relationship
        - content: The stable information
        - supporting_spans: List of original message IDs that support this schema
        
        Output Format:
        Return a JSON object with a single key "facts". 
        The value of "facts" must be a list of STRINGS, where each string is a valid JSON representation of the schema.

        Example format:
        {
            "facts": [
                "{\\"type\\": \\"preference\\", \\"content\\": \\"Alice prefers sci-fi movies over thrillers\\", \\"supporting_spans\\": [\\"msg_123\\", \\"msg_124\\"]}"
            ]
        }
        """

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "l2_schema_memory", 
                    "host": "localhost", 
                    "port": 6333, 
                    "embedding_model_dims": EMBEDDING_SIZE
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model, 
                    "openai_base_url": llm_base_url, 
                    "api_key": llm_api_key,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": False
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
                    "embedding_dims": EMBEDDING_SIZE,
                },
            },
            "custom_fact_extraction_prompt": self.schema_prompt
        }
        self.mem0 = Memory.from_config(config)

    def add_schema_memory(self, messages, user_id, metadata=None):
        """存储模式/事实记忆"""
        if metadata is None:
            metadata = {}
        
        # 为模式记忆添加元数据
        metadata["layer"] = "L2"
        metadata["type"] = "schema"
        
        logger.info(f"Adding schema memory for user_id={user_id}")
        try:
            result = self.mem0.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
                infer=True
            )
            logger.debug(f"Schema memory added successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding schema memory: {e}")
            raise

    def search_schema_memory(self, query, user_id, limit=100):
        """搜索模式记忆"""
        logger.info(f"Searching schema memory for user_id={user_id}, query='{query}'")
        try:
            results = self.mem0.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            logger.debug(f"Found {len(results) if isinstance(results, list) else 'N/A'} results")
            return results
        except Exception as e:
            logger.error(f"Error searching schema memory: {e}")
            raise

    def get_schema_memory_by_id(self, memory_id):
        """根据ID获取模式记忆"""
        logger.debug(f"Getting schema memory by id={memory_id}")
        try:
            memory = self.mem0.get(memory_id)
            return memory
        except Exception as e:
            logger.error(f"Error getting schema memory by id: {e}")
            raise

    def add_schema_memory_batch(self, messages_list, user_ids, metadata_list=None):
        """批量添加模式记忆"""
        logger.info(f"Batch adding schema memory for {len(messages_list)} items")
        results = []
        for i, messages in enumerate(messages_list):
            user_id = user_ids[i]
            metadata = metadata_list[i] if metadata_list else None
            try:
                result = self.add_schema_memory(messages, user_id, metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch add at index {i}: {e}")
                # decide whether to continue or raise. Usually batch operations might want to continue or fail hard.
                # Here we raise to be consistent with previous behavior (which would crash on error)
                raise
        return results
