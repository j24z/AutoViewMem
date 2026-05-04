import os
import json
import math
import concurrent.futures
from openai import OpenAI
from .l0_raw_memory import L0RawMemory
from .l1_event_frame import L1EventFrame
from .l2_schema_memory import L2SchemaMemory
from ..config import ARK_MODEL, ARK_BASE_URL, ARK_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS

class ResolutionPolicy:
    def __init__(self):
        self.client = OpenAI(
            base_url=ARK_BASE_URL,
            api_key=ARK_API_KEY
        )
        
        # 初始化各层记忆
        self.l0 = L0RawMemory()
        self.l1 = L1EventFrame()
        self.l2 = L2SchemaMemory()
        
        # 升级后的 Prompt：不再做单选，而是做权重分析
        self.policy_prompt = """
        Analyze the user's question to determine how to distribute search attention across memory layers.
        
        # Memory Layers Definitions:
        - **L0 (Raw)**: Exact quotes, specific wording, recent dialogue context. Use for "What did he say exactly?".
        - **L1 (Events)**: Timelines, sequence of actions, cause-and-effect, episode summaries. Use for "What happened yesterday?".
        - **L2 (Schema)**: Static facts, user profiles, preferences, relationships. Use for "What implies does Alice like?".

        # Task:
        Assign a relevance score (0-10) to each layer based on the question. 
        - 0 means completely irrelevant.
        - 10 means critical for answering.
        
        # Response Format (JSON only):
        {
            "L0_score": 3,
            "L1_score": 8,
            "L2_score": 5,
            "reasoning": "The question asks about a sequence of events (L1 high) and some facts (L2 medium), but exact quotes are less needed."
        }
        """

    def _get_layer_distribution(self, query, total_limit):
        """
        调用 LLM 获取各层权重，并计算具体的 limit 分配
        """
        try:
            response = self.client.chat.completions.create(
                model=ARK_MODEL,
                messages=[
                    {"role": "system", "content": self.policy_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            # 获取分数，默认为 5 (均等)
            s0 = result.get("L0_score", 5)
            s1 = result.get("L1_score", 5)
            s2 = result.get("L2_score", 5)
            
            total_score = s0 + s1 + s2
            if total_score == 0: total_score = 1 # 防止除以零

            # 按比例分配 limit
            # 至少保留 1 个名额给每一层（除非显式为0），防止完全漏掉某些偶然信息
            l0_limit = math.floor((s0 / total_score) * total_limit)
            l1_limit = math.floor((s1 / total_score) * total_limit)
            l2_limit = total_limit - l0_limit - l1_limit # 剩余的给 L2，确保总和等于 total_limit
            
            # 简单的边界修正：如果分数 > 0 但计算结果为 0，强制给 1（从最多的那个扣）
            # 这里简化处理，直接返回
            return {
                "L0": max(0, l0_limit),
                "L1": max(0, l1_limit),
                "L2": max(0, l2_limit),
                "scores": result
            }

        except Exception as e:
            print(f"Router Error: {e}, using default balanced distribution.")
            # 发生错误时的回退策略：均分
            avg = total_limit // 3
            return {
                "L0": avg,
                "L1": avg, 
                "L2": total_limit - avg*2,
                "scores": {"error": str(e)}
            }

    def search(self, query, user_id, total_limit=30):
        """
        根据问题动态分配检索配额，并并行执行检索
        """
        # 1. 获取分配策略
        distribution = self._get_layer_distribution(query, total_limit)
        
        limits = {
            "L0": distribution["L0"],
            "L1": distribution["L1"],
            "L2": distribution["L2"]
        }
        
        results = {}
        
        # 2. 并行检索 (关键优化：因为现在要查多次 IO，串行会很慢)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_l0 = None
            future_l1 = None
            future_l2 = None

            # 只有当分配了配额才去查
            if limits["L0"] > 0:
                future_l0 = executor.submit(self.l0.search_raw_messages, query, user_id, limits["L0"])
            
            if limits["L1"] > 0:
                future_l1 = executor.submit(self.l1.search_event_frames, query, user_id, limits["L1"])
                
            if limits["L2"] > 0:
                future_l2 = executor.submit(self.l2.search_schema_memory, query, user_id, limits["L2"])

            # 获取结果
            results["L0"] = future_l0.result() if future_l0 else []
            results["L1"] = future_l1.result() if future_l1 else []
            results["L2"] = future_l2.result() if future_l2 else []

        # 3. 结果合并与返回
        # 这里可以选择将结果拍平，或者保持层级结构供 Answer Prompt 使用
        # 建议保持结构，因为 Answer Prompt 需要区分层级
        
        return {
            "policy_decision": distribution, # 返回决策过程供调试
            "results": results, # {"L0": [...], "L1": [...], "L2": [...]}
            "total_retrieved": len(results["L0"]) + len(results["L1"]) + len(results["L2"])
        }

    def add_all_layers(self, messages, user_id, metadata=None):
        """将消息添加到所有三层记忆 (保持原样，也可以加并行优化)"""
        # 写入通常不需要像读取那样极低的延迟，但并行写入也是好的实践
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f0 = executor.submit(self.l0.add_raw_message, messages, user_id, metadata)
            f1 = executor.submit(self.l1.add_event_frame, messages, user_id, metadata)
            f2 = executor.submit(self.l2.add_schema_memory, messages, user_id, metadata)
            
            return {
                "l0": f0.result(),
                "l1": f1.result(),
                "l2": f2.result()
            }
