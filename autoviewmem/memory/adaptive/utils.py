import time
import os
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from autoviewmem.config import LLM_TEMPERATURE, LLM_MAX_TOKENS


def _print_llm_response(response):
    try:
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
        reasoning_content = getattr(message, "reasoning_content", "") or ""

        print("\n===== LLM Response =====")
        print("--- reasoning_content ---")
        print(reasoning_content if reasoning_content else "<empty>")
        print("--- content ---")
        print(content if content else "<empty>")
        print("===== End LLM Response =====\n")
    except Exception as e:
        print(f"[LLM DEBUG] Failed to print response: {e}")

def call_llm_with_retry(client_method, max_retries=10, base_delay=2, **kwargs):
    """
    Wraps an LLM API call with retry logic for common errors.
    
    Args:
        client_method: The callable API method (e.g., client.chat.completions.create)
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Initial delay in seconds for exponential backoff.
        **kwargs: Arguments to pass to the client_method.
        
    Returns:
        The result of the API call.
        
    Raises:
        The last exception caught if all retries fail.
    """
    delay = base_delay
    last_exception = None
    
    # Set default parameters if not provided
    if "temperature" not in kwargs:
        kwargs["temperature"] = LLM_TEMPERATURE
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = LLM_MAX_TOKENS
        
    # Preserve the caller's LLM request parameters. Do not force thinking mode here.
    extra_body = kwargs.get("extra_body")
    if extra_body is not None and "chat_template_kwargs" in extra_body:
        kwargs["extra_body"] = extra_body
    
    for attempt in range(max_retries):
        try:
            response = client_method(**kwargs)
            _print_llm_response(response)
            return response
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            last_exception = e
            if attempt == max_retries - 1:
                print(f"LLM API request failed after {max_retries} attempts: {e}")
                raise e
            
            # Add some jitter to avoid thundering herd if running in parallel
            import random
            sleep_time = delay + random.uniform(0, 1)
            
            print(f"LLM API request failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            delay *= 2  # Exponential backoff
        except Exception as e:
            # For other exceptions (e.g. 400 Bad Request), don't retry
            print(f"LLM API request failed with non-retriable error: {e}")
            raise e
            
    if last_exception:
        raise last_exception
