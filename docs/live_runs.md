# Живі прогони /reformulate

2026-09-30 14:47, 5 запусків на ціль, пауза 20 с, сервіс: `{"status": "ok", "db": "ok", "llm_provider": "gemini", "llm_fallback_provider": "groq"}`.

| Ціль | Успішних | Середня тривалість | Помилки | Закваску замінено | Провайдер | LLM-викликів (min–max) | Токенів у середньому |
|---|---|---|---|---|---|---|---|
| remove_allergen | 5/5 | 5.2 с | — | 5/5 | gemini | 2.0 (2–2) | 4400 |
| reduce_sugar | 5/5 | 5.0 с | — | — | gemini | 2.2 (2–3) | 5600 |
| make_vegan | 5/5 | 5.3 с | — | 5/5 | gemini | 2.0 (2–2) | 4447 |

## Усі запуски

| # | Ціль | HTTP | Результат | Тривалість | Провайдер | LLM-викликів | Токенів | run_id |
|---|---|---|---|---|---|---|---|---|
| 1 | remove_allergen | 200 | OK | 3.9 с | gemini | 2 | 4291 | 135 |
| 1 | reduce_sugar | 200 | OK | 6.2 с | gemini | 3 | 9122 | 136 |
| 1 | make_vegan | 200 | OK | 4.0 с | gemini | 2 | 4266 | 137 |
| 2 | remove_allergen | 200 | OK | 11.3 с | gemini | 2 | 4063 | 138 |
| 2 | reduce_sugar | 200 | OK | 3.8 с | gemini | 2 | 4689 | 139 |
| 2 | make_vegan | 200 | OK | 4.9 с | gemini | 2 | 4271 | 140 |
| 3 | remove_allergen | 200 | OK | 3.7 с | gemini | 2 | 4743 | 141 |
| 3 | reduce_sugar | 200 | OK | 6.1 с | gemini | 2 | 4767 | 142 |
| 3 | make_vegan | 200 | OK | 9.4 с | gemini | 2 | 4906 | 143 |
| 4 | remove_allergen | 200 | OK | 3.5 с | gemini | 2 | 4718 | 144 |
| 4 | reduce_sugar | 200 | OK | 4.8 с | gemini | 2 | 4918 | 145 |
| 4 | make_vegan | 200 | OK | 3.7 с | gemini | 2 | 3945 | 146 |
| 5 | remove_allergen | 200 | OK | 3.6 с | gemini | 2 | 4186 | 147 |
| 5 | reduce_sugar | 200 | OK | 4.1 с | gemini | 2 | 4505 | 148 |
| 5 | make_vegan | 200 | OK | 4.3 с | gemini | 2 | 4849 | 149 |
