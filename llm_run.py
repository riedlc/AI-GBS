
import asyncio
import random
from openai import AsyncOpenAI

_openai_client = None

async def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key="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",
        )
    return _openai_client

async def cleanup_clients():
    """Clean up async clients"""
    global _openai_client
    if _openai_client:
        await _openai_client.close()
        _openai_client = None

async def chat(model, prompt, system_prompt=None, temperature=0.0, max_tokens=None, max_retries=3):
    messages = []
    
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    
    # Retry logic for API failures
    for attempt in range(max_retries):
        try:
            client = await get_openai_client()
            
            kwargs = {
                'model': model,
                'messages': messages,
                'temperature': temperature
            }
            if max_tokens is not None:
                kwargs['max_tokens'] = max_tokens
                
            response = await client.chat.completions.create(**kwargs)
            return response
                
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                raise

# Usage:
if __name__ == "__main__":
    async def test():
        # OpenAI
        result = await chat('gpt-4o-mini', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
        print("OpenAI:", result)
    
    asyncio.run(test())