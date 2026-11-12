import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from prompt_manager import build_arc_prompt, build_code_prompt, build_reflection_prompt, parse_model_response, parse_code_response
from llm_client import generate_solution
from validator import resolve_task_path, validate_task
from code_executor import test_code_on_train_examples, execute_transform_safely


def run_single_task(
    task_identifier: str,
    output_dir: Path,
    mode: str = "end-to-end",
    provider: str = "google",
    model: str | None = None,
    reasoning_level: str = "high",
    samples: int = 1,
    temperature: float = 0.0,
) -> dict:
    task_path = resolve_task_path(task_identifier)
    if not task_path.exists():
        raise FileNotFoundError(f"Arquivo da tarefa não encontrado: '{task_identifier}'")

    task_name = task_path.stem
    task_out_dir = output_dir / task_name
    task_out_dir.mkdir(parents=True, exist_ok=True)

    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    start_time = time.time()
    train_passed = False
    train_score = (0, len(task_data.get("train", [])))
    total_usage = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_attempts = 0
    used_model = model

    if mode == "code-execution":
        prompt = build_code_prompt(task_data)
        raw_response, used_model, usage = generate_solution(
            prompt=prompt,
            provider=provider,
            model_name=model,
            reasoning_level=reasoning_level,
            temperature=temperature,
        )
        total_attempts += 1
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        code_string = parse_code_response(raw_response)
        train_passed, correct_train, total_train, _ = test_code_on_train_examples(
            code_string, task_data.get("train", [])
        )
        train_score = (correct_train, total_train)

        test_outputs = []
        for test_ex in task_data.get("test", []):
            try:
                out = execute_transform_safely(code_string, test_ex["input"])
                test_outputs.append(out)
            except Exception:
                test_outputs.append([])

        answer_data = {
            "rule_summary": f"Code generation mode. Train passed: {correct_train}/{total_train}",
            "test_outputs": test_outputs,
            "raw_code": code_string,
        }
    elif mode == "reflection":
        # Pass@k / Iterative Reflection Loop (up to 3 attempts per sample)
        best_code = None
        best_correct = -1
        final_test_outputs = []
        max_attempts_per_sample = 3

        for sample_idx in range(samples):
            if sample_idx > 0:
                time.sleep(1.0)

            current_prompt = build_code_prompt(task_data)
            current_code = ""
            passed_train = False

            for attempt in range(max_attempts_per_sample):
                total_attempts += 1
                if attempt > 0:
                    time.sleep(1.0)

                raw_response, used_model, usage = generate_solution(
                    prompt=current_prompt,
                    provider=provider,
                    model_name=model,
                    reasoning_level=reasoning_level,
                    temperature=max(0.2, temperature) if sample_idx > 0 or attempt > 0 else 0.0,
                )
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)

                current_code = parse_code_response(raw_response)
                passed_train, correct_train, total_train, error_msg = test_code_on_train_examples(
                    current_code, task_data.get("train", [])
                )

                if correct_train > best_correct:
                    best_correct = correct_train
                    best_code = current_code
                    train_score = (correct_train, total_train)
                    train_passed = passed_train

                if passed_train:
                    break

                # Prepare reflection prompt with error feedback
                current_prompt = build_reflection_prompt(task_data, current_code, error_msg)

            if passed_train:
                break

        # Execute best code on test
        test_outputs = []
        if best_code:
            for test_ex in task_data.get("test", []):
                try:
                    out = execute_transform_safely(best_code, test_ex["input"])
                    test_outputs.append(out)
                except Exception:
                    test_outputs.append([])

        answer_data = {
            "rule_summary": f"Reflection mode (samples={samples}). Best train score: {train_score[0]}/{train_score[1]}",
            "test_outputs": test_outputs,
            "raw_code": best_code,
        }
    else:
        prompt = build_arc_prompt(task_data)
        raw_response, used_model, usage = generate_solution(
            prompt=prompt,
            provider=provider,
            model_name=model,
            reasoning_level=reasoning_level,
            temperature=temperature,
        )
        total_attempts += 1
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        try:
            answer_data = parse_model_response(raw_response)
        except Exception as e:
            answer_data = {
                "error": f"Falha ao interpretar JSON da resposta: {e}",
                "rule_summary": "N/A",
                "test_outputs": [],
                "raw_response": raw_response,
            }

    duration = time.time() - start_time
    is_correct, rule_summary = validate_task(task_path, answer_data, verbose=False)

    prompt_file = task_out_dir / "prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt if mode != "reflection" else "Reflection mode prompt sequence")

    log_file = task_out_dir / "log.json"
    log_data = {
        "task_name": task_name,
        "task_file": str(task_path),
        "mode": mode,
        "provider": provider,
        "model": used_model,
        "reasoning_level": reasoning_level,
        "samples": samples,
        "total_attempts": total_attempts,
        "duration_seconds": round(duration, 4),
        "usage": total_usage,
        "is_correct": is_correct,
        "train_passed": train_passed if mode in ["code-execution", "reflection"] else None,
        "train_score": train_score if mode in ["code-execution", "reflection"] else None,
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    answer_file = task_out_dir / "answer.json"
    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, indent=2, ensure_ascii=False)

    status_str = "RESOLVED (CORRECT)" if is_correct else "FAILED"

    print(f"\n[{task_name}] Mode: {mode}")
    print(f"  Status:           {status_str}")
    if mode in ["code-execution", "reflection"]:
        print(f"  Train Accuracy:   {train_score[0]}/{train_score[1]}")
        print(f"  Total Attempts:   {total_attempts}")
    print(f"  Input Tokens:     {total_usage.get('input_tokens', 0)}")
    print(f"  Reasoning Tokens: {total_usage.get('reasoning_tokens', 0)}")
    print(f"  Answer Tokens:    {total_usage.get('output_tokens', 0)}")
    print(f"  Duration:         {duration:.2f}s")
    print(f"  Saved to:         {task_out_dir}")

    return {
        "task_name": task_name,
        "task_file": str(task_path),
        "is_correct": is_correct,
        "train_passed": train_passed if mode in ["code-execution", "reflection"] else None,
        "train_score": train_score if mode in ["code-execution", "reflection"] else None,
        "rule_summary": rule_summary,
        "total_attempts": total_attempts,
        "duration_seconds": round(duration, 4),
        "usage": total_usage,
        "artifacts_dir": str(task_out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Solver Nível 3")
    parser.add_argument("task_or_file", help="Tarefa JSON ou arquivo .txt com lista de tarefas")
    parser.add_argument(
        "--mode",
        choices=["end-to-end", "code-execution", "reflection"],
        default="end-to-end",
        help="Abordagem a ser utilizada (padrão: end-to-end)",
    )
    parser.add_argument("--output-dir", default="runs", help="Diretório base de execuções")
    parser.add_argument("--provider", default="google", help="Provedor de LLM")
    parser.add_argument("--model", default=None, help="Nome do modelo")
    parser.add_argument("--reasoning", default="high", help="Nível de raciocínio")
    parser.add_argument("--samples", type=int, default=1, help="Número de amostras Pass@k (modo reflection)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura de amostragem")
    parser.add_argument("--label", default="experiment", help="Rótulo identificador")
    args = parser.parse_args()

    input_target = Path(args.task_or_file)
    is_batch = input_target.suffix.lower() == ".txt" and input_target.exists()

    base_output_dir = Path(args.output_dir) / args.label
    base_output_dir.mkdir(parents=True, exist_ok=True)

    resultados_dir = Path("resultados")
    resultados_dir.mkdir(parents=True, exist_ok=True)

    if is_batch:
        with open(input_target, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        print(f"Iniciando experimento '{args.label}' [Modo: {args.mode}] com {len(tasks)} tarefa(s).")
        print("Pausa de 1.0 segundo entre as chamadas à API.")

        results = []
        for idx, task_id in enumerate(tasks):
            print(f"\n--- Processando tarefa {idx + 1}/{len(tasks)}: {task_id} ---")
            try:
                res = run_single_task(
                    task_identifier=task_id,
                    output_dir=base_output_dir,
                    mode=args.mode,
                    provider=args.provider,
                    model=args.model,
                    reasoning_level=args.reasoning,
                    samples=args.samples,
                    temperature=args.temperature,
                )
                results.append(res)
            except Exception as e:
                print(f"Erro ao processar a tarefa '{task_id}': {e}")
                results.append({
                    "task_name": task_id,
                    "is_correct": False,
                    "error": str(e),
                })

            if idx < len(tasks) - 1:
                time.sleep(1.0)

        total_tasks = len(results)
        solved_count = sum(1 for r in results if r.get("is_correct"))
        train_100_count = sum(1 for r in results if r.get("train_passed") is True)
        accuracy = (solved_count / total_tasks * 100) if total_tasks > 0 else 0.0

        total_input_tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
        total_reasoning_tokens = sum(r.get("usage", {}).get("reasoning_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
        total_duration = sum(r.get("duration_seconds", 0.0) for r in results)
        total_attempts_sum = sum(r.get("total_attempts", 1) for r in results)

        summary = {
            "experiment_label": args.label,
            "mode": args.mode,
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "reasoning_level": args.reasoning,
            "samples": args.samples,
            "tasks_file": str(input_target),
            "total_tasks": total_tasks,
            "solved_tasks": solved_count,
            "train_100_tasks": train_100_count,
            "accuracy_percentage": round(accuracy, 2),
            "total_attempts": total_attempts_sum,
            "total_input_tokens": total_input_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration_seconds": round(total_duration, 2),
            "results": results,
        }

        if args.mode == "reflection":
            summary_filename = "reflexao_summary.json"
        elif args.mode == "code-execution":
            summary_filename = "geracao_codigo_summary.json"
        else:
            summary_filename = "fim_a_fim_summary.json"

        summary_file = resultados_dir / summary_filename
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n========================================")
        print(f"EXPERIMENTO CONCLUÍDO [{args.mode}]: {args.label}")
        print(f"Tarefas Resolvidas: {solved_count}/{total_tasks} ({accuracy:.1f}%)")
        if args.mode in ["code-execution", "reflection"]:
            print(f"100% no Treino: {train_100_count}/{total_tasks}")
        print(f"Resumo salvo em: {summary_file}")
        print("========================================")

    else:
        run_single_task(
            task_identifier=args.task_or_file,
            output_dir=base_output_dir,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            reasoning_level=args.reasoning,
            samples=args.samples,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
