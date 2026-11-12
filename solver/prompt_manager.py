import json


def build_arc_prompt(task_data: dict) -> str:
    """
    Constrói um prompt didático e estruturado para o desafio ARC-AGI (End-to-End).
    """
    prompt = """You are an expert AI system solving the ARC-AGI (Abstraction and Reasoning Corpus) challenge.
Your goal is to discover the underlying transformation rule from input grid(s) to output grid(s) shown in the training demonstrations, and apply that rule to predict the output grid(s) for the test input(s).

Colors/Symbols in ARC grids are represented by integers from 0 to 9:
0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: magenta, 7: orange, 8: cyan, 9: brown.

Below are the training examples:
"""
    for idx, example in enumerate(task_data.get("train", [])):
        prompt += f"\n--- Training Example {idx + 1} ---\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"
        prompt += f"Output:\n{json.dumps(example['output'])}\n"

    prompt += "\nBelow are the test inputs for which you must predict the output:\n"
    for idx, example in enumerate(task_data.get("test", [])):
        prompt += f"\n--- Test Input {idx + 1} ---\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"

    prompt += """
Please analyze the patterns carefully:
1. Examine spatial properties (shapes, lines, symmetries, colors, bounding boxes, counts).
2. Infer the transformation rule that maps inputs to outputs consistently.
3. Formulate the predicted output grid for each test case.

Return your response strictly in the following JSON format without Markdown code blocks:
{
  "rule_summary": "<Explain concisely the identified transformation rule>",
  "test_outputs": [
    [
      [<int>, <int>, ...],
      [<int>, <int>, ...]
    ]
  ]
}
Note: "test_outputs" must be a list of 2D grids (one 2D grid for each test input in the exact order).
"""
    return prompt


def build_code_prompt(task_data: dict) -> str:
    """
    Constrói um prompt para a Abordagem B (Geração de Código Python).
    """
    prompt = """You are an expert Python programmer solving the ARC-AGI challenge.
Write a Python function named `transform(grid: list[list[int]]) -> list[list[int]]` that implements the transformation rule observed in the training examples.

Colors/Symbols in ARC grids are represented by integers from 0 to 9:
0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: magenta, 7: orange, 8: cyan, 9: brown.

Below are the training examples:
"""
    for idx, example in enumerate(task_data.get("train", [])):
        prompt += f"\n--- Training Example {idx + 1} ---\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"
        prompt += f"Output:\n{json.dumps(example['output'])}\n"

    prompt += """
Provide your solution inside a Python code block ```python ... ``` containing the `transform` function and a brief explanation in a comment if needed.
"""
    return prompt


def build_reflection_prompt(task_data: dict, previous_code: str, error_message: str) -> str:
    """
    Constrói um prompt de reflexão contendo o código anterior e o erro detectado na validação de treino.
    """
    prompt = """You are an expert Python programmer debugging an ARC-AGI challenge solution.
Your previous Python implementation failed when tested against the training examples.

Below are the training examples:
"""
    for idx, example in enumerate(task_data.get("train", [])):
        prompt += f"\n--- Training Example {idx + 1} ---\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"
        prompt += f"Output:\n{json.dumps(example['output'])}\n"

    prompt += f"""
Your previous code:
```python
{previous_code}
```

Error encountered during execution or validation:
{error_message}

Please analyze why the code failed, correct the logic of the `transform(grid: list[list[int]]) -> list[list[int]]` function, and provide the corrected Python solution inside a ```python ... ``` code block.
"""
    return prompt


def parse_model_response(raw_text: str) -> dict:
    """
    Extrai e decodifica o JSON retornado pelo modelo (End-to-End).
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return json.loads(cleaned[start_idx : end_idx + 1])
        raise


def parse_code_response(raw_text: str) -> str:
    """
    Extrai o código Python da resposta do modelo.
    """
    cleaned = raw_text.strip()
    if "```python" in cleaned:
        parts = cleaned.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0].strip()
            return code_part
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            code_part = parts[1].strip()
            return code_part
    return cleaned
