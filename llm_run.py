# llm_run.py - Updated with async and better error handling
# from ollama import AsyncClient as OllamaAsyncClient
try:
    from openai import AsyncOpenAI
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")
    AsyncOpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv package not installed. Install with: pip install python-dotenv")
    load_dotenv = lambda: None

import os
import asyncio
import random

# Load environment variables
load_dotenv()

# Global client instances to reuse connections
_ollama_client = None
_openai_client = None

async def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaAsyncClient()
    return _ollama_client

async def get_openai_client():
    global _openai_client
    if AsyncOpenAI is None:
        raise ImportError("openai package not installed. Install with: pip install openai")
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key="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",
        )
    return _openai_client

async def cleanup_clients():
    """Clean up async clients"""
    global _ollama_client, _openai_client
    if _openai_client:
        await _openai_client.close()
        _openai_client = None
    # Ollama client doesn't need explicit cleanup

async def chat(model, prompt, system_prompt=None, temperature=0.0, client_type="ollama", max_tokens=None, max_retries=3):
    messages = []
    
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    
    # Retry logic for API failures
    for attempt in range(max_retries):
        try:
            if client_type == "ollama":
                client = await get_ollama_client()
                options = {'temperature': temperature}
                if max_tokens is not None:
                    options['num_predict'] = max_tokens
                    
                response = await client.chat(
                    model=model, 
                    messages=messages,
                    options=options
                )
                return response
                
            elif client_type == "openai":
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
                print(f"All {max_retries} attempts failed for {client_type}")
                raise
    
    raise ValueError(f"Unknown client_type: {client_type}")

# Usage:
if __name__ == "__main__":
    async def test():
        # Ollama
        result = await chat('llama3.1:8b', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
        print("Ollama:", result)
        
        # OpenAI
        result = await chat('gpt-4o-mini', 'Generate a number between 1 and 100', temperature=0, client_type="openai", max_tokens=5)
        print("OpenAI:", result)
    
    asyncio.run(test())