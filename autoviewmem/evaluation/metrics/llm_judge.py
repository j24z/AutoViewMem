import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from openai import OpenAI

from autoviewmem.config import LLM_TEMPERATURE, LLM_MAX_TOKENS
# from config import MODEL_NAME

# 从环境变量读取模型配置，提供合理默认值
from dotenv import load_dotenv  # 若未安装，可按需去掉并自行加载
load_dotenv()
ARK_API_KEY = os.getenv("ARK_API_KEY", "any")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "http://localhost:8001/v1")
ARK_MODEL = os.getenv("ARK_MODEL", "1")

MODEL_NAME = ARK_MODEL
client = OpenAI(
    base_url=ARK_BASE_URL,
    api_key=ARK_API_KEY,
)


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return your evaluation in JSON format with the following keys:
- "reasoning": a short (one sentence) explanation of your reasoning.
- "label": either "CORRECT" or "WRONG".

Do NOT include any other text before or after the JSON.
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    max_retries = 20
    delay = 10  # 3 minutes

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": ACCURACY_PROMPT.format(
                            question=question, gold_answer=gold_answer, generated_answer=generated_answer
                        ),
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                #response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            # print(raw)
            
            # More robust JSON extraction: find the first '{' and last '}'
            try:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1:
                    text = raw[start:end+1]
                else:
                    text = extract_json(raw)
            except Exception:
                text = extract_json(raw)

            try:
                data = json.loads(text)
            except Exception as e:
                # If standard parsing fails, try a very simple regex for label
                import re
                label_match = re.search(r'"label":\s*"(\w+)"', raw, re.IGNORECASE)
                if label_match:
                    data = {"label": label_match.group(1)}
                else:
                    print(f"⚠️ JSON parse error (Attempt {attempt+1}/{max_retries}), raw response:")
                    print(raw)
                    print("Extracted:", text)
                    print("Error:", e)
                    # Raise error to trigger retry
                    raise ValueError(f"JSON parsing failed: {e}")

            # 强制安全检查
            label = data.get("label", None)
            if label is None:
                print(f"⚠️ Missing 'label' in JSON (Attempt {attempt+1}/{max_retries}). Raw:")
                print(raw)
                print("JSON:", data)
                raise ValueError("Missing 'label' in JSON")

            return 1 if label.strip().upper() == "CORRECT" else 0

        except Exception as e:
            print(f"Error in evaluate_llm_judge (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning 0.")
                return 0
