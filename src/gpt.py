import os
import time
import openai
import asyncio
import aiolimiter
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio


async def _throttled_openai_completion_acreate(
    model,
    prompt,
    temperature,
    max_tokens,
    top_p,
    n_samples,
    limiter,
    max_attempts=10,
    initial_delay=1,
    exponential_base=2,
    max_delay=30,
):
    if prompt is None:
        return None
    async with limiter:
        attempt = 0
        while max_attempts is None or attempt < max_attempts:
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n_samples,
                )
            except openai.error.RateLimitError:
                await asyncio.sleep(min(max_delay, initial_delay * (exponential_base ** attempt)))
            except openai.OpenAIError as e:
                print(f"Attempt {attempt}: {e}.")
                await asyncio.sleep(min(max_delay, initial_delay * (exponential_base ** attempt)))
            attempt += 1 
        print(f"Max Attempts.")
        return None


async def async_dispatch_openai_requests(
    requests: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n_samples: int,
    requests_per_minute: int = 1000,
) -> list[str]:
    if "OPENAI_ORG_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_ORG_KEY environment variable must be set when using OpenAI API."
        )
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.organization = os.environ["OPENAI_ORG_KEY"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)

    async_responses = [
        _throttled_openai_completion_acreate(
            model=model,
            prompt=req,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_samples=n_samples,
            limiter=limiter,
        )
        for req in requests
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return responses


def _throttled_openai_completion_create(
    model,
    prompt,
    temperature,
    max_tokens,
    top_p,
    n_samples,
    max_attempts=None,
    initial_delay=1,
    exponential_base=2,
):
    if prompt is None:
        return None
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        try:
            return openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n_samples,
            )
        except openai.error.RateLimitError:
            time.sleep(initial_delay * (exponential_base ** attempt))
        except openai.OpenAIError as e:
            print(f"Attempt {attempt}: {e}.")
            time.sleep(initial_delay * (exponential_base ** attempt))
        attempt += 1

def dispatch_openai_requests(
    requests: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n_samples: int,
    requests_per_minute: int = 1000,
) -> list[str]:
    if "OPENAI_ORG_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_ORG_KEY environment variable must be set when using OpenAI API."
        )
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.organization = os.environ["OPENAI_ORG_KEY"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    responses = [
        _throttled_openai_completion_create(
            model=model,
            prompt=req,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_samples=n_samples,
        )
        for req in requests
    ]
    return responses

def run_gpt(
    requests,
    model='gpt-4',
    temperature=0.0,
    max_tokens=512,
    top_p=1.0,
    n_samples=1,
    run_async=True
):
    assert (n_samples == 1) != (temperature > 0.0)
    if run_async:
        responses = asyncio.run(
            async_dispatch_openai_requests(
                requests=requests,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n_samples=n_samples,
            )
        )
    else:
        responses = dispatch_openai_requests(
            requests=requests,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_samples=n_samples,
        )
    return list(responses)

def make_messages(messages, system_prompt=None):
    if type(messages) == str:
        return make_messages([messages])
    if messages is None or any(m is None for m in messages):
        return None
    output = [
        {
            "role": "system",
            "content": system_prompt or "You are a helpful assistant."
        }
    ]
    for i, m in enumerate(messages):
        role = "user" if i % 2 else "assistant"
        output.append({
            "role": role,
            "content": m,
        })
    return output
