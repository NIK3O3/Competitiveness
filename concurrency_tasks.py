from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")

counter = 0
counter_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def timer() -> float:
    """Повертає час із високою точністю."""
    return time.perf_counter()


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_two_threads(target: Callable[..., None], *args: Any) -> None:
    """Запускає два потоки з однаковою функцією."""
    thread_1 = threading.Thread(target=target, args=args)
    thread_2 = threading.Thread(target=target, args=args)

    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()


def chunks_from_iterable(data: Iterable[T], chunk_size: int) -> Iterator[list[T]]:
    """Розбиває iterable на списки фіксованого розміру."""
    iterator = iter(data)

    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def chunked_range(start: int, stop: int, chunk_size: int) -> Iterator[range]:
    """Розбиває range на кілька range-частин без створення великого списку."""
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        yield range(chunk_start, chunk_stop)


# ---------------------------------------------------------------------------
# Завдання 1. Race condition
# ---------------------------------------------------------------------------

def increment_racy(iterations: int) -> None:
    """
    Небезпечне збільшення глобального counter.

    Умова з counter += 1 логічно є операцією read-modify-write:
    1. прочитати counter;
    2. додати 1;
    3. записати нове значення.

    Для наочної демонстрації race condition ми спеціально розкладаємо
    операцію на кроки й іноді віддаємо керування іншому потоку через sleep(0).
    """
    global counter

    for i in range(iterations):
        current_value = counter

        # Штучно збільшуємо шанс перемикання потоку саме між read і write.
        if i % 100 == 0:
            time.sleep(0)

        counter = current_value + 1


def task_1_race_condition(iterations: int = 100_000, repeats: int = 5) -> None:
    print_header("Завдання 1. Race condition")

    expected = iterations * 2

    for attempt in range(1, repeats + 1):
        global counter
        counter = 0

        run_two_threads(increment_racy, iterations)

        print(f"Спроба {attempt}: counter = {counter}; очікувалось = {expected}")

    print(
        "\nПояснення:\n"
        "- Результат неправильний, бо два потоки одночасно читають і змінюють спільну "
        "глобальну змінну counter.\n"
        "- Race condition — це ситуація, коли результат програми залежить від порядку "
        "виконання потоків.\n"
        "- counter += 1 не є атомарною логічною операцією: між читанням і записом "
        "інший потік може змінити counter."
    )


# ---------------------------------------------------------------------------
# Завдання 2. Усунення проблеми через Lock
# ---------------------------------------------------------------------------

def increment_with_lock(iterations: int) -> None:
    """Безпечне збільшення counter через Lock."""
    global counter

    for i in range(iterations):
        with counter_lock:
            current_value = counter

            # Навіть із примусовим перемиканням потоку результат буде правильним,
            # бо критична секція захищена lock.
            if i % 100 == 0:
                time.sleep(0)

            counter = current_value + 1


def task_2_lock(iterations: int = 100_000, repeats: int = 3) -> None:
    print_header("Завдання 2. Усунення проблеми через Lock")

    expected = iterations * 2

    for attempt in range(1, repeats + 1):
        global counter
        counter = 0

        run_two_threads(increment_with_lock, iterations)

        print(f"Спроба {attempt}: counter = {counter}; очікувалось = {expected}")

    print(
        "\nПояснення:\n"
        "- Lock працює, бо дозволяє тільки одному потоку одночасно входити в критичну "
        "секцію.\n"
        "- Критична секція тут — читання counter, обчислення нового значення і запис назад.\n"
        "- Мінуси: повільніше виконання, блокування потоків, ризик deadlock при неправильному "
        "використанні, складніший код."
    )


# ---------------------------------------------------------------------------
# Завдання 3. Без mutable state
# ---------------------------------------------------------------------------

def increment(x: int) -> int:
    """Чиста функція: не змінює глобальний стан, а повертає нове значення."""
    return x + 1


def task_3_without_mutable_state() -> None:
    print_header("Завдання 3. Без mutable state")

    values = [1, 2, 3, 4, 5]
    result = list(map(increment, values))

    print(f"Вхідні значення: {values}")
    print(f"Результат:       {result}")
    print(
        "\nПояснення:\n"
        "- Глобальних змінних немає.\n"
        "- Функція increment(x) не має side effects.\n"
        "- Кожне значення обробляється незалежно, тому race condition тут немає."
    )


# ---------------------------------------------------------------------------
# Завдання 4. Паралельна обробка без стану
# ---------------------------------------------------------------------------

def square(x: int) -> int:
    return x * x


def task_4_parallel_square() -> None:
    print_header("Завдання 4. Паралельна обробка без стану")

    data = [1, 2, 3, 4, 5]

    with ThreadPoolExecutor() as executor:
        result = list(executor.map(square, data))

    print(f"data = {data}")
    print(f"Квадрати = {result}")


# ---------------------------------------------------------------------------
# Завдання 5. Паралельний map
# ---------------------------------------------------------------------------

def parallel_map(
    func: Callable[[T], R],
    data: Iterable[T],
    max_workers: int | None = None,
    chunk_size: int = 1_000,
) -> list[R]:
    """
    Паралельний map на базі ThreadPoolExecutor.

    Особливість:
    - дані обробляються chunk-ами, щоб не створювати мільйон futures для великого списку;
    - порядок результатів зберігається.
    """
    def apply_chunk(chunk: list[T]) -> list[R]:
        return [func(item) for item in chunk]

    result: list[R] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_result in executor.map(apply_chunk, chunks_from_iterable(data, chunk_size)):
            result.extend(chunk_result)

    return result


def task_5_parallel_map() -> None:
    print_header("Завдання 5. Паралельний map")

    data = [1, 2, 3, 4, 5]
    result = parallel_map(square, data, chunk_size=2)

    print(f"data = {data}")
    print(f"parallel_map(square, data) = {result}")


# ---------------------------------------------------------------------------
# Завдання 6. Порівняння часу
# ---------------------------------------------------------------------------

def task_6_time_comparison(size: int = 1_000_000) -> None:
    print_header("Завдання 6. Порівняння часу")

    data = range(size)

    start = timer()
    normal_result = list(map(square, data))
    normal_time = timer() - start

    start = timer()
    parallel_result = parallel_map(square, range(size), chunk_size=10_000)
    parallel_time = timer() - start

    print(f"Кількість елементів: {size:,}".replace(",", " "))
    print(f"Звичайний map: {normal_time:.4f} сек.")
    print(f"parallel_map:  {parallel_time:.4f} сек.")
    print(f"Результати однакові: {normal_result == parallel_result}")

    print(
        "\nВисновок:\n"
        "- Для дуже простої CPU-операції x*x ThreadPool може бути повільнішим за звичайний map.\n"
        "- Причини: overhead на потоки, планування задач і GIL у CPython.\n"
        "- ThreadPool корисніший для I/O-bound задач, а не для легких CPU-bound операцій."
    )


# ---------------------------------------------------------------------------
# Завдання 7. CPU-bound задача
# ---------------------------------------------------------------------------

def heavy_task(x: int) -> int:
    total = 0
    for i in range(10_000_000):
        total += i * x
    return total


def task_7_cpu_bound() -> None:
    print_header("Завдання 7. CPU-bound задача")

    data = [1, 2, 3, 4]
    workers = min(4, os.cpu_count() or 1)

    start = timer()
    sequential_result = [heavy_task(x) for x in data]
    sequential_time = timer() - start

    start = timer()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        thread_result = list(executor.map(heavy_task, data))
    thread_time = timer() - start

    start = timer()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        process_result = list(executor.map(heavy_task, data))
    process_time = timer() - start

    print(f"Дані: {data}")
    print(f"Кількість worker-ів: {workers}")
    print(f"Послідовно:  {sequential_time:.4f} сек.")
    print(f"ThreadPool:  {thread_time:.4f} сек.")
    print(f"ProcessPool: {process_time:.4f} сек.")
    print(f"Результати однакові: {sequential_result == thread_result == process_result}")

    print(
        "\nВисновок:\n"
        "- Для CPU-bound задач ProcessPool зазвичай швидший, бо використовує окремі процеси.\n"
        "- ThreadPool у CPython часто не прискорює CPU-bound код через GIL.\n"
        "- ProcessPool має overhead на створення процесів і передачу даних, тому для малих задач "
        "він може бути неефективним."
    )


# ---------------------------------------------------------------------------
# Завдання 8. Паралельний pipeline
# ---------------------------------------------------------------------------

def greater_than_100(x: int) -> bool:
    return x > 100


def task_8_parallel_pipeline() -> None:
    print_header("Завдання 8. Паралельний pipeline")

    data = range(100)

    # map -> x*x
    squares = parallel_map(square, data, chunk_size=10)

    # filter -> x > 100
    flags = parallel_map(greater_than_100, squares, chunk_size=10)
    filtered = [value for value, keep in zip(squares, flags) if keep]

    # reduce -> sum
    result = reduce(lambda acc, item: acc + item, filtered, 0)

    print("Pipeline: map(x*x) -> filter(x > 100) -> reduce(sum)")
    print(f"Результат = {result}")


# ---------------------------------------------------------------------------
# Завдання 9. Functional pipeline API
# ---------------------------------------------------------------------------

def pipeline(data: Iterable[Any], steps: Sequence[Callable[[Iterable[Any]], Any]]) -> Any:
    """
    Універсальний functional pipeline.

    steps — список функцій. Кожна функція отримує результат попереднього кроку.
    """
    result: Any = data

    for step in steps:
        result = step(result)

    return result


def step_parallel_square(data: Iterable[int]) -> list[int]:
    return parallel_map(square, data, chunk_size=10)


def step_filter_gt_100(data: Iterable[int]) -> filter:
    return filter(greater_than_100, data)


def step_sum(data: Iterable[int]) -> int:
    return reduce(lambda acc, item: acc + item, data, 0)


def task_9_functional_pipeline_api() -> None:
    print_header("Завдання 9. Functional pipeline API")

    data = range(100)
    steps = [
        step_parallel_square,
        step_filter_gt_100,
        step_sum,
    ]

    result = pipeline(data, steps)

    print("pipeline(data, [parallel_square, filter_gt_100, sum])")
    print(f"Результат = {result}")


# ---------------------------------------------------------------------------
# Завдання 10. Safe execution
# ---------------------------------------------------------------------------

def risky(x: int) -> float:
    if x == 0:
        raise ValueError("x не може дорівнювати 0")
    return 10 / x


def safe_call(func: Callable[[T], R], item: T) -> tuple[bool, R | None, str | None]:
    """Виконує функцію без падіння всієї програми."""
    try:
        return True, func(item), None
    except Exception as error:
        return False, None, f"{type(error).__name__}: {error}"


def safe_parallel_map(
    func: Callable[[T], R],
    data: Iterable[T],
    max_workers: int | None = None,
) -> list[tuple[bool, R | None, str | None]]:
    """
    Паралельний map із перехопленням помилок.

    Повертає список:
    - (True, value, None) — якщо виконання успішне;
    - (False, None, "ErrorType: message") — якщо була помилка.
    """
    items = list(data)
    results: list[tuple[bool, R | None, str | None]] = [(False, None, None)] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(safe_call, func, item): index
            for index, item in enumerate(items)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


def task_10_safe_execution() -> None:
    print_header("Завдання 10. Safe execution")

    data = [5, 2, 1, 0, -1, 10]
    result = safe_parallel_map(risky, data)

    print(f"data = {data}")
    print("Результат safe_parallel_map:")

    for item, output in zip(data, result):
        ok, value, error = output
        if ok:
            print(f"x = {item:>3} -> OK: {value}")
        else:
            print(f"x = {item:>3} -> ERROR: {error}")

    print("\nПрограма не падає, бо помилки перехоплюються всередині safe_call().")


# ---------------------------------------------------------------------------
# Завдання 11. Обробка транзакцій
# ---------------------------------------------------------------------------

def transaction_filter(x: int) -> bool:
    """Приклад filter: беремо тільки парні транзакції."""
    return x % 2 == 0


def transaction_map(x: int) -> int:
    """Приклад map: перетворюємо транзакцію."""
    return x * 2


def process_transaction_chunk(chunk: range) -> int:
    """
    Один chunk транзакцій:
    filter -> map -> sum.

    Функція винесена на верхній рівень, щоб її можна було використовувати
    в ProcessPoolExecutor.
    """
    return sum(transaction_map(x) for x in chunk if transaction_filter(x))


def parallel_transaction_pipeline(
    total_transactions: int = 1_000_000,
    chunk_size: int = 50_000,
    max_workers: int | None = None,
) -> int:
    chunks = chunked_range(0, total_transactions, chunk_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        partial_sums = executor.map(process_transaction_chunk, chunks)

    return sum(partial_sums)


def task_11_transactions(total_transactions: int = 1_000_000) -> None:
    print_header("Завдання 11. Обробка транзакцій")

    workers = min(4, os.cpu_count() or 1)

    start = timer()
    result = parallel_transaction_pipeline(
        total_transactions=total_transactions,
        chunk_size=50_000,
        max_workers=workers,
    )
    elapsed = timer() - start

    print("Pipeline: filter -> map -> sum")
    print("filter: x % 2 == 0")
    print("map:    x * 2")
    print(f"Кількість транзакцій: {total_transactions:,}".replace(",", " "))
    print(f"Результат: {result}")
    print(f"Час: {elapsed:.4f} сек.")


# ---------------------------------------------------------------------------
# Завдання 12. API simulation
# ---------------------------------------------------------------------------

def fetch_data(x: int) -> int:
    time.sleep(1)
    return x


def task_12_api_simulation(count: int = 10) -> None:
    print_header("Завдання 12. API simulation")

    data = list(range(count))

    start = timer()
    sequential_result = [fetch_data(x) for x in data]
    sequential_time = timer() - start

    start = timer()
    with ThreadPoolExecutor(max_workers=count) as executor:
        parallel_result = list(executor.map(fetch_data, data))
    parallel_time = timer() - start

    print(f"data = {data}")
    print(f"Послідовно: {sequential_result}; час = {sequential_time:.4f} сек.")
    print(f"Паралельно: {parallel_result}; час = {parallel_time:.4f} сек.")

    print(
        "\nВисновок:\n"
        "- fetch_data імітує I/O-bound задачу, бо більшість часу йде на очікування sleep(1).\n"
        "- ThreadPoolExecutor добре підходить для таких задач.\n"
        "- 10 запитів послідовно займають приблизно 10 секунд, паралельно — приблизно 1 секунду."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_selected_task(task_number: str, quick: bool) -> None:
    iterations = 10_000 if quick else 100_000
    map_size = 100_000 if quick else 1_000_000
    transaction_count = 100_000 if quick else 1_000_000

    tasks: dict[str, Callable[[], None]] = {
        "1": lambda: task_1_race_condition(iterations=iterations),
        "2": lambda: task_2_lock(iterations=iterations),
        "3": task_3_without_mutable_state,
        "4": task_4_parallel_square,
        "5": task_5_parallel_map,
        "6": lambda: task_6_time_comparison(size=map_size),
        "7": task_7_cpu_bound,
        "8": task_8_parallel_pipeline,
        "9": task_9_functional_pipeline_api,
        "10": task_10_safe_execution,
        "11": lambda: task_11_transactions(total_transactions=transaction_count),
        "12": task_12_api_simulation,
    }

    if task_number == "all":
        for number in map(str, range(1, 13)):
            tasks[number]()
        return

    if task_number not in tasks:
        available = ", ".join(["all"] + list(tasks.keys()))
        raise ValueError(f"Невідоме завдання: {task_number}. Доступні варіанти: {available}")

    tasks[task_number]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Завдання з конкурентності у Python")
    parser.add_argument(
        "--task",
        default="all",
        help="Номер завдання: 1..12 або all. За замовчуванням: all",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Швидший режим для завдань 1, 2, 6, 11",
    )

    args = parser.parse_args()
    run_selected_task(task_number=args.task, quick=args.quick)


if __name__ == "__main__":
    # Потрібно для коректної роботи ProcessPoolExecutor на Windows.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
