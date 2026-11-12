import concurrent.futures


def _execute_code(code_string: str, grid: list[list[int]]) -> list[list[int]]:
    local_vars = {}
    exec(code_string, {}, local_vars)
    if "transform" not in local_vars:
        raise ValueError("A função 'transform' não foi encontrada no código gerado.")
    transform_func = local_vars["transform"]
    return transform_func(grid)


def execute_transform_safely(code_string: str, grid: list[list[int]], timeout_seconds: float = 3.0) -> list[list[int]]:
    """
    Executa a função transform gerada pelo LLM com um limite de tempo (timeout).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_code, code_string, grid)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"A execução excedeu o limite de {timeout_seconds} segundos.")
        except Exception as e:
            raise RuntimeError(f"Erro ao executar o código: {e}")


def test_code_on_train_examples(code_string: str, train_examples: list[dict]) -> tuple[bool, int, int, str]:
    """
    Testa o código gerado em todos os exemplos de treino.
    Retorna: (todos_acertaram, total_acertos, total_exemplos, mensagem_erro)
    """
    total = len(train_examples)
    correct_count = 0
    error_msg = ""

    for idx, example in enumerate(train_examples):
        inp = example["input"]
        expected = example["output"]
        try:
            output = execute_transform_safely(code_string, inp, timeout_seconds=3.0)
            if output == expected:
                correct_count += 1
            else:
                error_msg = f"Training example {idx + 1} failed. Expected {expected}, got {output}."
                break
        except Exception as e:
            error_msg = f"Training example {idx + 1} raised exception: {e}."
            break

    return correct_count == total, correct_count, total, error_msg
