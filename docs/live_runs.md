# Живі прогони /reformulate

2026-09-30 13:10, 5 запусків на ціль, пауза 20 с, сервіс: `{"status": "ok", "db": "ok", "llm_provider": "gemini", "llm_fallback_provider": "groq"}`.

| Ціль | Успішних | Середня тривалість | Помилки | Закваску замінено | Провайдер | LLM-викликів (min–max) | Токенів у середньому |
|---|---|---|---|---|---|---|---|
| remove_allergen | 5/5 | 6.9 с | — | 5/5 | gemini | 2.0 (2–2) | 4453 |
| reduce_sugar | 5/5 | 7.9 с | — | — | gemini | 2.2 (2–3) | 5539 |
| make_vegan | 5/5 | 4.5 с | — | 5/5 | gemini | 2.0 (2–2) | 4339 |

## Усі запуски

| # | Ціль | HTTP | Результат | Тривалість | Провайдер | LLM-викликів | Токенів | run_id |
|---|---|---|---|---|---|---|---|---|
| 1 | remove_allergen | 200 | OK | 3.7 с | gemini | 2 | 4435 | 105 |
| 1 | reduce_sugar | 200 | OK | 16.0 с | gemini | 2 | 4706 | 106 |
| 1 | make_vegan | 200 | OK | 4.1 с | gemini | 2 | 4521 | 107 |
| 2 | remove_allergen | 200 | OK | 18.0 с | gemini | 2 | 4556 | 108 |
| 2 | reduce_sugar | 200 | OK | 12.3 с | gemini | 3 | 9359 | 109 |
| 2 | make_vegan | 200 | OK | 7.2 с | gemini | 2 | 4492 | 110 |
| 3 | remove_allergen | 200 | OK | 4.2 с | gemini | 2 | 4546 | 111 |
| 3 | reduce_sugar | 200 | OK | 4.7 с | gemini | 2 | 4789 | 112 |
| 3 | make_vegan | 200 | OK | 4.2 с | gemini | 2 | 4498 | 113 |
| 4 | remove_allergen | 200 | OK | 4.2 с | gemini | 2 | 4659 | 114 |
| 4 | reduce_sugar | 200 | OK | 3.5 с | gemini | 2 | 4439 | 115 |
| 4 | make_vegan | 200 | OK | 3.7 с | gemini | 2 | 4118 | 116 |
| 5 | remove_allergen | 200 | OK | 4.4 с | gemini | 2 | 4069 | 117 |
| 5 | reduce_sugar | 200 | OK | 2.9 с | gemini | 2 | 4400 | 118 |
| 5 | make_vegan | 200 | OK | 3.5 с | gemini | 2 | 4068 | 119 |
