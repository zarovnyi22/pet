# Блок 1 — Скелет (орієнтир: 1 година)

## Промпт для Claude Code

```
Прочитай CLAUDE.md повністю.

Зроби тільки блок "Скелет", більше нічого:
- pyproject.toml через uv з базовими залежностями (fastapi, uvicorn, asyncpg,
  pydantic-settings, sentence-transformers)
- FastAPI застосунок з GET /health, який повертає {"status": "ok", "db": "...", "llm_provider": "..."}
- docker-compose.yml з двома сервісами: api і db (образ pgvector/pgvector:pg16), volume для
  даних БД, healthcheck через pg_isready, api стартує тільки після здорової бази
- Dockerfile двостадійний (uv ставить залежності на першій стадії)
- migrations/001_init.sql з трьома таблицями (documents, chunks з pgvector, reformulation_runs)
  і застосуй цю міграцію автоматично при старті застосунку
- .env.example з усіма змінними з коментарями (DATABASE_URL, LLM_PROVIDER, GEMINI_API_KEY,
  GROQ_API_KEY, EMBEDDING_MODEL, AGENT_MAX_ITERATIONS, AGENT_TIMEOUT_SECONDS), реальний .env
  в .gitignore

Зупинись після цього блоку, я сам перевірю контрольну точку.

Коли скажу, що контрольна точка пройдена — закоммить усе з повідомленням "Block 1: skeleton"
і запуш у `origin main` (репозиторій вже підключений).
```

## Перевір, перш ніж йти далі

1. `cp .env.example .env`
2. `docker compose up --build`
3. В окремому терміналі: `curl http://localhost:8000/health`

Очікується відповідь із `"status": "ok"` і `"db"` не з помилкою — тобто застосунок реально
достукався до бази, а не просто сам собі відповів.

Якщо не піднялось за ~30 секунд — це вже відхилення від нефункціональної вимоги проєкту,
розбирайся з Claude Code одразу, не переходь далі.
