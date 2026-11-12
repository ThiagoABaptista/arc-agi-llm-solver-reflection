import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GOOGLE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.5-flash-lite"

BUDGET_MAP = {
    "minimal": 128,
    "low": 512,
    "medium": 1024,
    "high": 2048,
}


def call_google_api(prompt: str, model_name: str, reasoning_level: str = "high", temperature: float = 0.0) -> tuple[str, dict]:
    """
    Executa a chamada usando a SDK oficial do Google GenAI com reasoning e temperature configurados.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not defined in the environment or .env file.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config_kwargs = {}

    if "gemini" in model_name.lower():
        budget = BUDGET_MAP.get(reasoning_level, 2048)
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

    if temperature > 0.0:
        config_kwargs["temperature"] = temperature

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    prompt_tokens = 0
    reasoning_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0

        details = getattr(usage, "candidates_token_details", None)
        if details:
            if isinstance(details, list):
                for item in details:
                    reasoning_tokens += getattr(item, "thinking_token_count", 0) or 0
            else:
                reasoning_tokens = getattr(details, "thinking_token_count", 0) or 0

        output_tokens = max(0, candidates_tokens - reasoning_tokens)

    token_usage = {
        "input_tokens": prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or (prompt_tokens + candidates_tokens),
    }

    return response.text, token_usage


def call_openrouter_api(prompt: str, model_name: str, reasoning_level: str = "high", temperature: float = 0.0) -> tuple[str, dict]:
    """
    Executa a chamada via OpenRouter API com esforço de reasoning e temperature definidos.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not defined in the environment or .env file.")

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    extra_body = {
        "reasoning": {
            "effort": reasoning_level
        }
    }

    completion_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "extra_body": extra_body,
    }

    if temperature > 0.0:
        completion_kwargs["temperature"] = temperature

    completion = client.chat.completions.create(**completion_kwargs)

    raw_text = completion.choices[0].message.content

    prompt_tokens = 0
    reasoning_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if hasattr(completion, "usage") and completion.usage is not None:
        usage = completion.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        output_tokens = max(0, completion_tokens - reasoning_tokens)

    token_usage = {
        "input_tokens": prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    return raw_text, token_usage


def generate_solution(
    prompt: str,
    provider: str = "google",
    model_name: str | None = None,
    reasoning_level: str = "high",
    temperature: float = 0.0,
) -> tuple[str, str, dict]:
    """
    Despacha a chamada para o provedor selecionado (google ou openrouter).
    """
    if provider == "google":
        model = model_name or DEFAULT_GOOGLE_MODEL
        raw_response, usage = call_google_api(prompt, model, reasoning_level=reasoning_level, temperature=temperature)
    elif provider == "openrouter":
        model = model_name or DEFAULT_OPENROUTER_MODEL
        raw_response, usage = call_openrouter_api(prompt, model, reasoning_level=reasoning_level, temperature=temperature)
    else:
        raise ValueError(f"Provedor desconhecido: '{provider}'. Escolha 'google' ou 'openrouter'.")

    return raw_response, model, usage
