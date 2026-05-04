import os
import json
from mem0 import Memory
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, EMBEDDING_NAME, EMBEDDING_SIZE, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..log_config import logger

class L1EventFrame:
    def __init__(self):
        logger.info("Initializing L1EventFrame...")
        llm_model = ARK_MODEL
        llm_base_url = ARK_BASE_URL
        llm_api_key = ARK_API_KEY

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "l1_event_frame", 
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
        
        # 不再使用自定义prompt，使用mem0默认的事实提取prompt
        pass

    def add_event_frame(self, messages, user_id, metadata=None):
        """存储事件框架"""
        if metadata is None:
            metadata = {}
        
        # 为事件框架添加元数据
        metadata["layer"] = "L1"
        metadata["type"] = "event_frame"
        
        logger.info(f"Adding event frame for user_id={user_id}")
        try:
            result = self.mem0.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
                infer=True
                # 不传入prompt，使用mem0默认的事实提取prompt
            )
            logger.debug(f"Event frame added successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding event frame: {e}")
            raise

    def search_event_frames(self, query, user_id, limit=100):
        """搜索事件框架"""
        logger.info(f"Searching event frames for user_id={user_id}, query='{query}'")
        try:
            results = self.mem0.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            logger.debug(f"Found {len(results) if isinstance(results, list) else 'N/A'} event frames")
            return results
        except Exception as e:
            logger.error(f"Error searching event frames: {e}")
            raise

    def get_event_frame_by_id(self, memory_id):
        """根据ID获取事件框架"""
        logger.debug(f"Getting event frame by id={memory_id}")
        try:
            memory = self.mem0.get(memory_id)
            return memory
        except Exception as e:
            logger.error(f"Error getting event frame by id: {e}")
            raise

    def add_event_frame_batch(self, messages_list, user_ids, metadata_list=None):
        """批量添加事件框架"""
        logger.info(f"Batch adding event frames for {len(messages_list)} items")
        results = []
        for i, messages in enumerate(messages_list):
            user_id = user_ids[i]
            metadata = metadata_list[i] if metadata_list else None
            try:
                result = self.add_event_frame(messages, user_id, metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch add event frame at index {i}: {e}")
                raise
        return results
