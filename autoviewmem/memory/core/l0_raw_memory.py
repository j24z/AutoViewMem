import os
import json
from mem0 import Memory
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, EMBEDDING_NAME, EMBEDDING_SIZE, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..log_config import logger

class L0RawMemory:
    def __init__(self):
        logger.info("Initializing L0RawMemory...")
        llm_model = ARK_MODEL
        llm_base_url = ARK_BASE_URL
        llm_api_key = ARK_API_KEY

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "l0_raw_memory", 
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
                "provider": "huggingface",
                "config": {"model": EMBEDDING_NAME},
            },
        }
        self.mem0 = Memory.from_config(config)

    def add_raw_message(self, messages, user_id, metadata=None):
        """存储原始消息片段"""
        if metadata is None:
            metadata = {}
        
        # 为每个消息添加L0层元数据
        for msg in messages:
            msg["layer"] = "L0"
            msg["provenance"] = {
                "type": "raw_message",
                "timestamp": metadata.get("timestamp", None),
                "speaker": metadata.get("speaker", None)
            }
        
        logger.info(f"Adding raw message for user_id={user_id}")
        try:
            result = self.mem0.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
                infer=False  # 不进行推理，直接存储原始内容
            )
            logger.debug(f"Raw message added successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding raw message: {e}")
            raise

    def search_raw_messages(self, query, user_id, limit=100):
        """搜索原始消息"""
        logger.info(f"Searching raw messages for user_id={user_id}, query='{query}'")
        try:
            results = self.mem0.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            logger.debug(f"Found {len(results) if isinstance(results, list) else 'N/A'} raw messages")
            return results
        except Exception as e:
            logger.error(f"Error searching raw messages: {e}")
            raise

    def get_raw_message_by_id(self, memory_id):
        """根据ID获取原始消息"""
        logger.debug(f"Getting raw message by id={memory_id}")
        try:
            memory = self.mem0.get(memory_id)
            return memory
        except Exception as e:
            logger.error(f"Error getting raw message by id: {e}")
            raise

    def add_raw_message_batch(self, messages_list, user_ids, metadata_list=None):
        """批量添加原始消息"""
        logger.info(f"Batch adding raw messages for {len(messages_list)} items")
        results = []
        for i, messages in enumerate(messages_list):
            user_id = user_ids[i]
            metadata = metadata_list[i] if metadata_list else None
            try:
                result = self.add_raw_message(messages, user_id, metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch add raw message at index {i}: {e}")
                raise
        return results
