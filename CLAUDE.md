# Reformulation Assistant — правила проєкту для Claude Code

Pet-проєкт під вакансію Backend & AI Engineer у харчовій R&D. Мета: за ~8 годин зібрати
робочий сервіс, який демонструє весь стек із вакансії (FastAPI, Postgres+pgvector, Docker,
LLM, RAG, агент з інструментами) і придатний для демо на співбесіді.

Головне правило роботи: Claude пише код, людина тримає архітектуру і перевіряє кожен блок
руками (курл, а не "повірити на слово"). На співбесіді запитають "чому так", а не "хто писав".

## Що будуємо

HTTP-сервіс: технолог подає рецептуру продукту і ціль зміни (прибрати алерген / зменшити
цукор / зробити веганським), сервіс повертає обґрунтовану пропозицію замін інгредієнтів із
джерелами і порівнянням нутрієнтів до/після. Три частини: база знань (Postgres+pgvector),
RAG-пошук (/ask), агент переформулювання (/reformulate) з tool calling.

Фронтенду немає — демо через Swagger UI (`/docs`).

## Технологічний стек (фіксований)

| Шар | Вибір |
|---|---|
| Мова і пакети | Python 3.12, `uv`, один `pyproject.toml` |
| API | FastAPI, Pydantic v2, uvicorn |
| База даних і вектори | PostgreSQL 16 + pgvector, образ `pgvector/pgvector:pg16` |
| Доступ до бази | asyncpg напряму, ручний SQL (без SQLAlchemy/Alembic — один день) |
| Ембединги | sentence-transformers, `all-MiniLM-L6-v2`, 384 виміри, локально на CPU |
| LLM, основний | Gemini API (Google AI Studio), сімейство Flash, `LLM_PROVIDER=gemini` |
| LLM, запасний | Groq free tier, `openai/gpt-oss-120b` (Llama на Groq більше недоступна), `LLM_PROVIDER=groq` |
| Дані про продукти | Open Food Facts API, без ключа |
| Контейнери | Docker, Docker Compose (обов'язково) |
| Тести | pytest, pytest-asyncio, httpx |
| Kubernetes / CI | kind, GitHub Actions — бонуси |

## Жорсткі правила (не порушувати)

- **Без фреймворків для агента.** Ніякого LangChain/LlamaIndex — свій цикл tool calling на
  ~50 рядків. За день фреймворк з'їсть більше часу на дебаг, ніж зекономить.
- **Один клас `LLMClient`**, два методи: `complete` і `complete_with_tools`. Провайдер
  обирається лише через змінну оточення `LLM_PROVIDER` (`gemini` / `groq`). Перемикання між
  провайдерами — заміна однієї змінної, без переписування коду.
- **Тести ніколи не викликають живий LLM.** Тільки `app/llm/fake.py` (`FakeLLM`), який
  повертає задані відповіді. Живий LLM спалює безкоштовний ліміт і падає випадково.
- **Формат помилок єдиний:** `{"error": {"code": "...", "message": "..."}}`.
- **Міграції — один SQL-файл** `migrations/001_init.sql`, застосовується на старті додатка.
  Alembic на один день не потрібен.
- **Один коміт на блок плану**, а не один коміт на весь день і не один на кожен файл.
- Секрети (`GEMINI_API_KEY`, `GROQ_API_KEY`) — тільки через `.env` (в `.gitignore`), в
  `.env.example` — лише назви змінних з коментарями, без реальних значень.

## Структура репозиторію

```
reformulation-assistant/
├── app/
│   ├── main.py            # FastAPI app, роутери, lifespan
│   ├── config.py          # Settings з pydantic-settings
│   ├── db.py               # пул asyncpg, міграція на старті
│   ├── schemas.py          # Pydantic-моделі запитів і відповідей
│   ├── embeddings.py       # обгортка sentence-transformers
│   ├── chunking.py         # чиста функція різання тексту
│   ├── retrieval.py        # векторний пошук у pgvector
│   ├── llm/
│   │   ├── base.py          # інтерфейс LLMClient
│   │   ├── gemini.py
│   │   ├── groq.py
│   │   └── fake.py          # для тестів
│   ├── agent/
│   │   ├── loop.py          # цикл tool calling
│   │   ├── tools.py         # три інструменти і їх JSON-схеми
│   │   └── prompts.py
│   ├── routers/
│   │   ├── documents.py
│   │   ├── ask.py
│   │   └── reformulate.py
│   └── ingest.py            # CLI для папки з корпусом
├── data/corpus/             # 20 markdown-документів
├── migrations/001_init.sql
├── tests/
├── k8s/                     # бонус
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── Makefile
└── README.md
```

## Схема бази (3 таблиці)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  doc_id      TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  doc_type    TEXT NOT NULL CHECK (doc_type IN ('ingredient_spec','trial_report','guideline')),
  content     TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT  NOT NULL,
  text        TEXT NOT NULL,
  embedding   VECTOR(384) NOT NULL
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE reformulation_runs (
  id          BIGSERIAL PRIMARY KEY,
  request     JSONB NOT NULL,
  response    JSONB,
  trace       JSONB NOT NULL DEFAULT '[]',
  status      TEXT NOT NULL,
  duration_ms INT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Повторний інгест того самого `doc_id` замінює старі чанки, не додає дублікати.
`reformulation_runs` логує кожен запуск агента разом із trace — це відповідь на співбесідне
питання "як ти б дебажив агента в продакшені".

## Endpoints (4)

| Метод і шлях | Призначення | Вхід | Вихід |
|---|---|---|---|
| GET /health | Живий сервіс і база | — | `{"status","db","llm_provider"}` |
| POST /documents | Інгест одного документа | `{doc_id, title, doc_type, content}` | `{doc_id, chunks_created}` |
| POST /ask | RAG-відповідь із цитатами | `{question, top_k: 5}` | `{answer, sources: [{doc_id, title, chunk_text, score}]}` |
| POST /reformulate | Агент переформулювання | рецептура + goal + goal_params | substitutions + nutrition + trace |

Чанкування: 200 токенів, перекриття 40, рахуються токенізатором самої `all-MiniLM-L6-v2`
(не словами): модель обрізає вхід на 256 токенах, тож більший чанк ембедер побачив би лише
частково. Текст чанка — зріз оригіналу за offset-ами токенів. `/ask`: якщо відповіді немає в чанках — чесне
"в базі знань немає даних", а не вигадка; запит без жодного заіндексованого документа → 409
`empty_knowledge_base`.

`goal` для `/reformulate` — одне з трьох: `remove_allergen` (параметр `allergen`, один із 14
алергенів ЄС), `reduce_sugar` (параметр `percent` 10–50), `make_vegan` (без параметрів).

## Агент

Звичайний цикл tool calling (не фреймворк): LLM отримує системний промпт, рецептуру, ціль і
опис трьох інструментів, викликає їх, поки не готова відповісти, повертає JSON за схемою.

**Три інструменти:**

| Інструмент | Аргументи | Повертає | Реалізація |
|---|---|---|---|
| `search_knowledge_base` | `query: str, top_k: int = 5` | чанки з doc_id/title/text/score | той самий векторний пошук, що й `/ask` |
| `lookup_product` | `name: str` | нутрієнти/100г, алергени, інгредієнти | Open Food Facts, timeout 5с, in-memory кеш на час запуску |
| `calc_nutrition` | `ingredients: list[{name, grams, nutrients_per_100g}]` | kcal/protein/fat/carbs/sugar на 100г | чиста арифметика на Python, без LLM і мережі |

`calc_nutrition` — інструмент, а не промпт, бо LLM погано рахує зважені середні.

**Обмеження циклу:** максимум 6 ітерацій і 60 секунд. Перевищено → 504 `agent_timeout` +
той trace, що встиг назбирати. Захист від зациклення: якщо модель два рази підряд викликає
той самий інструмент з тими самими аргументами — цикл примусово просить фінальну відповідь.

**Системний промпт, 4 правила:**
1. Спочатку база знань, потім Open Food Facts. Внутрішні спеки пріоритетніші.
2. Кожна заміна посилається на `doc_id` або продукт з Open Food Facts. Без джерела →
   `"confidence": "low"` і запис у `warnings`.
3. Нутрієнти до/після — тільки через `calc_nutrition`, ніколи вручну.
4. Фінальна відповідь — тільки JSON за схемою, без тексту навколо.

Невалідний JSON від LLM → один повтор із текстом помилки, максимум 1 раз. Кожен виклик
інструмента і кожна помилка валідації потрапляють у `trace`.

**Запасний варіант, якщо агент не стабілізується вчасно (орієнтир — 5:00 від старту):**
замість вільного циклу — фіксований пайплайн з трьох кроків, де код сам викликає інструменти
в заданому порядку, а LLM лише формулює пошуковий запит і фінальну відповідь. Якщо
довелось піти цим шляхом — чесно написати про це в README як свідомий компроміс.

## Дані для RAG

20 markdown-документів у `data/corpus/` (генеруються окремим запитом до Claude, не руками):
12 `ingredient_spec` (функція інгредієнта, дозування, алергени, нутрієнти/100г, чим
замінити), 6 `trial_report` (звіт з пробної варки), 2 `guideline` (політика з алергенів,
зниження цукру). Інгредієнти мають покривати всі три цілі агента: молоко і рослинні
аналоги, цукор і підсолоджувачі, яйце і замінники, пшеничне борошно і безглютенові суміші.
Кожен документ 200–500 слів, з frontmatter `doc_id, title, doc_type`. Варто переглянути 2–3
документи очима — якщо там немає конкретних чисел і замін, RAG-у нічого буде цитувати.

## Тести (мінімальний набір, без ключів і без інтернету)

| Тест | Що перевіряє | Залежності |
|---|---|---|
| `test_chunking.py` | розмір чанків, перекриття, порожній текст | жодних |
| `test_calc_nutrition.py` | зважене середнє/100г, нульова маса, відсутні нутрієнти | жодних |
| `test_agent_loop.py` | FakeLLM: виклик інструменту, зупинка на ліміті, повтор при невалідному JSON, trace | FakeLLM |
| `test_api.py` | `/health`, `/documents`, `/ask` через `httpx.AsyncClient` з FakeLLM і фейковим ембедером | Postgres через docker compose, `pytest.mark.skipif` якщо нема `DATABASE_URL` |

`ruff check` і `ruff format --check` мають бути чисті.

## Нефункціональні вимоги

`docker compose up --build` на чистій машині піднімає сервіс без ручних кроків (крім
`.env.example` → `.env` і вставки ключа LLM), і через 30 секунд `/health` відповідає.
Двостадійний Dockerfile: перша стадія — `uv` ставить залежності, друга — лише код і venv.
Модель ембедингів завантажується на етапі збірки образу, не при старті контейнера.

## Логи

Структурні JSON-логи через стандартний `logging`, один форматтер. Кожен запит отримує
`request_id`, який прокидається в логи агента.

## Бонуси (тільки після готовності основного блоку)

1. Kubernetes на kind (`k8s/`: Namespace, Secret, Deployment+Service з liveness/readiness на
   `/health`, StatefulSet+Service для Postgres з PVC). Без Helm.
2. GitHub Actions: Postgres як service container, `uv`, `ruff` + `pytest`, білд образу без
   push. Бедж у README.
3. Гібридний пошук: повнотекстовий `tsvector` + Reciprocal Rank Fusion з векторним пошуком.
4. Оцінка якості RAG: `eval/questions.jsonl` (10 пар питання/doc_id) + скрипт recall@5.
5. Хмара — тільки якщо вже є акаунт без картки; інакше абзац у README "як би я деплоїв".
