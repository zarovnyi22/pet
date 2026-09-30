# Живі прогони /reformulate

2026-09-30 12:09, 5 запусків на ціль, пауза 20 с, сервіс: `{"status": "ok", "db": "ok", "llm_provider": "gemini", "llm_fallback_provider": "groq"}`.

| Ціль | Успішних | Середня тривалість | Помилки | Закваску замінено | Провайдер | LLM-викликів (min–max) | Токенів у середньому |
|---|---|---|---|---|---|---|---|
| remove_allergen | 5/5 | 4.1 с | — | 5/5 | gemini | 2.0 (2–2) | 4570 |
| reduce_sugar | 5/5 | 4.3 с | — | — | gemini | 2.2 (2–3) | 5836 |
| make_vegan | 5/5 | 4.2 с | — | 5/5 | gemini | 2.0 (2–2) | 4462 |

## Усі запуски

| # | Ціль | HTTP | Результат | Тривалість | Провайдер | LLM-викликів | Токенів | run_id |
|---|---|---|---|---|---|---|---|---|
| 1 | remove_allergen | 200 | OK | 4.0 с | gemini | 2 | 4493 | 89 |
| 1 | reduce_sugar | 200 | OK | 3.9 с | gemini | 2 | 4578 | 90 |
| 1 | make_vegan | 200 | OK | 4.2 с | gemini | 2 | 4981 | 91 |
| 2 | remove_allergen | 200 | OK | 4.1 с | gemini | 2 | 4726 | 92 |
| 2 | reduce_sugar | 200 | OK | 4.2 с | gemini | 2 | 5281 | 93 |
| 2 | make_vegan | 200 | OK | 4.2 с | gemini | 2 | 3967 | 94 |
| 3 | remove_allergen | 200 | OK | 4.2 с | gemini | 2 | 4550 | 95 |
| 3 | reduce_sugar | 200 | OK | 3.5 с | gemini | 2 | 5157 | 96 |
| 3 | make_vegan | 200 | OK | 4.1 с | gemini | 2 | 4388 | 97 |
| 4 | remove_allergen | 200 | OK | 4.1 с | gemini | 2 | 4666 | 98 |
| 4 | reduce_sugar | 200 | OK | 6.1 с | gemini | 3 | 9355 | 99 |
| 4 | make_vegan | 200 | OK | 4.5 с | gemini | 2 | 4426 | 100 |
| 5 | remove_allergen | 200 | OK | 3.9 с | gemini | 2 | 4416 | 101 |
| 5 | reduce_sugar | 200 | OK | 3.9 с | gemini | 2 | 4809 | 102 |
| 5 | make_vegan | 200 | OK | 3.9 с | gemini | 2 | 4547 | 103 |
