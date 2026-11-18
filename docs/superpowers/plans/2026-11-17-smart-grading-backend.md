# 智能阅卷后端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有题库和静态复核页之上，实现 FastAPI + LangGraph + PostgreSQL 的智能阅卷后端，覆盖批改、人工复核、自然语言校准、学情报告和 Langfuse 观测。

**Architecture:** FastAPI 提供服务入口，LangGraph 顶层图负责装载、并行逐题批改、汇总、复核中断和报告；三个子图分别做题目结构、评分细则、错因归类。业务数据写 PostgreSQL，图状态用 Postgres checkpointer；测试用内存 checkpointer 和假 LLM。

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x async, psycopg 3, LangGraph, langgraph-checkpoint-postgres, OpenAI SDK（DeepSeek 兼容接口）, Langfuse, pytest, httpx, ruff。

**Spec:** `docs/superpowers/specs/2026-11-17-smart-grading-design.md`

## Global Constraints

- Python 3.12，虚拟环境 `.venv`。首次安装先运行 `.venv\Scripts\python.exe -m ensurepip --upgrade`，再运行 `.venv\Scripts\python.exe -m pip install -e ".[dev]"`。
- 数据库：开发库 `Mark`，测试库 `Mark_test`。测试绝不写 `Mark`。测试库在 Docker 容器 `agent_test-postgres` 里。
- 环境变量从 `.env` 读：`API_KEY`、`BASE_URL`、`MODEL_NAME`、`LANGFUSE_*`；业务库用 `MARK_DATABASE_URL`，默认 `postgresql+psycopg://postgres:postgres@localhost:5432/Mark`。
- 不提交 `.env`、API key、数据库密码。`.gitignore` 已经忽略 `.env`。
- 选择题默认 3 分，主观题默认 6 分。原始题库没有分值，这是假设。
- 复核阈值 `0.75`；低于阈值、0/满分且答案少于 20 字、JSON 修复过、细则兜底、空白答案，都要进复核。
- 错因 code 只用这九个：`info_filter`、`concept_misread`、`incomplete_summary`、`argument_analysis`、`evidence_misuse`、`expression`、`off_topic`、`blank`、`mcq_wrong`。
- 同一份 submission 的当前结果取最新 `status='completed'` 的 grading_run；历史 run 保留。
- `grading_runs.heartbeat_at` 由节点更新；启动清理超过 30 分钟没有心跳的 `running` 运行。
- 复核只做一轮。重复复核请求返回 409。
- 本期不做 LoRA、vLLM、OCR、React/Vue 前端。静态页是 `app/static/index.html`，可以继续用 mock 数据。
- 面向用户的文案和 prompt 用中文；代码标识符用英文。
- 不要自动 `git commit` 或 `git push`，除非用户明确要求。用户要求提交时，每条 commit message 末尾加：
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`

## Review Focus

这五类输入/条件最容易让实现者踩坑，每个都在对应任务的测试里固定下来：

1. 主观题空白答案：应得 0 分、错因 `blank`、进入复核，不能抛异常。
2. 选择题答案非法（空、多个选项、选项不在 A-D）：应得 0 分并进入复核，不能抛异常。
3. LLM 返回的 JSON 合法但内容非法（分数加总不对、point_id 不在细则里）：应重试一次，仍失败就兜底并标记复核，不能写入非法结果。
4. 同一份答卷有多个 grading_run（重复批改或校准重批）：submission、报告必须取最新 completed run，不能混用旧结果。
5. 重复提交复核或服务重启后恢复：不能重复应用同一条修正，不能把已完成的 run 改回 await_review。

## File Map

```
app/
  __init__.py
  main.py                       create_app、lifespan、路由、静态页
  core/config.py                Settings
  core/logging.py               基础日志配置
  db/models.py                  SQLAlchemy 模型
  db/session.py                 engine/session factory
  db/repositories.py            查询和写入函数
  db/seed.py                    错因分类、默认分值
  api/deps.py                   依赖注入
  api/routers/exams.py
  api/routers/submissions.py
  api/routers/grading.py
  api/routers/review.py
  api/routers/calibration.py
  api/routers/reports.py
  api/routers/dev.py
  schemas/exam.py
  schemas/submission.py
  schemas/grading.py
  schemas/rubric.py
  schemas/report.py
  graph/state.py                TypedDict 状态
  graph/builder.py              顶层图
  graph/checkpointer.py         AsyncPostgresSaver 生命周期
  graph/nodes/                  顶层节点
  graph/subgraphs/structure.py
  graph/subgraphs/rubric.py
  graph/subgraphs/error_analysis.py
  graph/subgraphs/grade_question.py
  services/llm.py
  services/observability.py
  services/grading.py
  services/calibration.py
  services/reports.py
  services/synthetic.py
  prompts/*.md
  static/index.html             已有
scripts/
  init_db.py
  init_scores.py
  seed_synthetic.py
  evaluate.py
tests/
  conftest.py
  fakes.py
  unit/
  integration/
  fixtures/golden_grading.jsonl
```

---

### Task 1: 项目脚手架、配置和健康检查

**Files:**
- Modify: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/core/__init__.py`
- Create: `app/core/config.py`
- Create: `app/main.py`
- Modify: `config.py`
- Create: `.env.example`
- Test: `tests/integration/test_health.py`

**Interfaces:**
- Consumes: 无。
- Produces: `Settings`（字段见下）、`settings` 单例、`create_app(settings: Settings | None = None) -> FastAPI`、`GET /health`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/integration/test_health.py
from httpx import ASGITransport, AsyncClient
from app.core.config import Settings
from app.main import create_app


async def test_health():
    app = create_app(Settings(_env_file=None, langfuse_enabled=False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_health.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app'` 或 `fastapi` 未安装。

- [ ] **Step 3: 配 `pyproject.toml` 和依赖**

`pyproject.toml` 增加：

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx", "ruff"]

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

`[project] dependencies` 写：`fastapi`、`uvicorn[standard]`、`sqlalchemy[asyncio]`、`psycopg[binary,pool]`、`pydantic-settings`、`langgraph`、`langgraph-checkpoint-postgres`、`langchain-core`、`openai`、`langfuse`。先不写版本号，安装后把 pip 解析出的版本回填。

安装：

```powershell
.venv\Scripts\python.exe -m ensurepip --upgrade
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 4: 写 `Settings` 和 `create_app`**

`app/core/config.py`：

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mark_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/Mark"
    test_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/Mark_test"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 2
    grading_concurrency: int = 5
    review_confidence_threshold: float = 0.75
    default_mcq_score: float = 3
    default_subjective_score: float = 6
    langfuse_enabled: bool = True
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    prompt_version: str = "v1"

settings = Settings()
```

`app/main.py`：

```python
from app.core.config import Settings, settings as default_settings

def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or default_settings
    app = FastAPI(title="Smart Grading", version="0.1.0")
    app.state.settings = resolved

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

app = create_app()
```

根目录 `config.py` 改成从 `app.core.config` 导出 `BASE_DIR`、`Settings`、`settings`。

`.env.example` 只放变量名和占位值，不放真实 key。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_health.py -v`
Expected: PASS。

- [ ] **Step 6: 静态检查**

Run: `.venv\Scripts\python.exe -m ruff check app tests`
Expected: 无错误。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add pyproject.toml app config.py .env.example tests
git commit -m "feat: scaffold FastAPI app and settings"
```

---

### Task 2: 应用数据库 schema、模型和测试库

**Files:**
- Modify: `db/schema.sql`
- Create: `app/db/__init__.py`
- Create: `app/db/models.py`
- Create: `app/db/session.py`
- Create: `tests/conftest.py`
- Test: `tests/integration/test_models.py`

**Interfaces:**
- Consumes: `Settings.mark_database_url`、`Settings.test_database_url`。
- Produces: `Base`、模型类 `Exam`、`Passage`、`Question`、`ImportRun`、`ClassGroup`、`Student`、`Submission`、`StudentAnswer`、`Rubric`、`GradingRun`、`GradingResult`、`TeacherCorrection`、`CalibrationLog`、`ErrorCategory`、`Report`；`create_engine_and_factory(url: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]`；pytest fixture `session`、`sample_exam`、`test_settings`、`test_session_factory`。

- [ ] **Step 1: 建测试库并写失败测试**

创建测试库：

```powershell
docker exec agent_test-postgres createdb -U postgres Mark_test
```

如果已经存在，命令报错可以忽略。

```python
# tests/integration/test_models.py
async def test_relationships(session, sample_exam):
    exam_id = sample_exam["exam_id"]
    student = Student(class_id=sample_exam["class_id"], name="测试学生", student_no="T001")
    session.add(student)
    await session.flush()
    submission = Submission(exam_id=exam_id, student_id=student.id)
    session.add(submission)
    await session.flush()
    question_id = sample_exam["questions"][0]["id"]
    session.add(StudentAnswer(submission_id=submission.id, question_id=question_id, selected_options=["C"]))
    await session.commit()

    loaded = await session.get(Submission, submission.id)
    assert loaded.student.name == "测试学生"
    assert len(loaded.answers) == 1
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_models.py -v`
Expected: FAIL，表或模型不存在。

- [ ] **Step 3: 扩展 `db/schema.sql`**

把设计文档 §7.1–§7.6 的 DDL 原样加进 `db/schema.sql`，并包含：

- `grading_runs.heartbeat_at TIMESTAMPTZ`
- `grading_results.knowledge_points JSONB NOT NULL DEFAULT '[]'::jsonb`
- `CREATE INDEX idx_grading_runs_submission ON grading_runs (submission_id, created_at DESC);`
- `CREATE UNIQUE INDEX uq_rubrics_active ON rubrics (question_id) WHERE active;`

保留现有 `exams`、`passages`、`questions`、`import_runs`。所有新表用 `CREATE TABLE IF NOT EXISTS`。

- [ ] **Step 4: 写模型和 session 工厂**

`app/db/models.py` 用 SQLAlchemy 2.0 的 `DeclarativeBase`、`Mapped`、`mapped_column`、`relationship`。表名和列名必须和 `db/schema.sql` 一致。`ClassGroup` 映射到表 `classes`。JSON 列用 `sqlalchemy.dialects.postgresql.JSONB`。金额/分数字段用 `Numeric(6, 1)` 或 `Numeric(5, 1)`。

`app/db/session.py`：

```python
def create_engine_and_factory(url: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory
```

- [ ] **Step 5: 写 `tests/conftest.py`**

必须提供：

- session 级 fixture：用 `docker exec -i agent_test-postgres psql -U postgres -d Mark_test -v ON_ERROR_STOP=1` 执行 `db/schema.sql`，然后 `TRUNCATE` 所有业务表和题库表并 `RESTART IDENTITY CASCADE`。
- `test_settings`：`Settings(_env_file=None, mark_database_url=..., langfuse_enabled=False)`。
- `settings`：`test_settings` 的别名，供不想写 `test_` 前缀的测试用。
- `test_session_factory`：测试库的 engine 和 session factory。
- `session`：每个测试一个 `AsyncSession`，测试结束后回滚。
- `sample_exam`：插入 1 个 exam、1 篇 passage、3 道题（Q1 四选一正确答案 C，Q2 四选一正确答案 A，Q3 主观题 3 个采分点，max_score 6），以及 1 个班和 1 个学生。返回 `{"exam_id", "exam_code", "class_id", "student_id", "questions": [{"id", "number", ...}]}`。
- `sample_submission`：在 `sample_exam` 基础上插入一个学生和一份答卷。
- 每次清表后插入 9 个 `error_categories`，避免测试依赖 `Mark` 库。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_models.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add db/schema.sql app/db tests/conftest.py tests/integration/test_models.py
git commit -m "feat: add grading schema and SQLAlchemy models"
```

---

### Task 3: 试卷读取 API

**Files:**
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/exam.py`
- Create: `app/db/repositories.py`
- Create: `app/api/__init__.py`
- Create: `app/api/deps.py`
- Create: `app/api/routers/__init__.py`
- Create: `app/api/routers/exams.py`
- Modify: `app/main.py`
- Test: `tests/integration/test_exam_api.py`

**Interfaces:**
- Consumes: `create_engine_and_factory`、模型。
- Produces: `GET /api/exams -> list[ExamSummary]`、`GET /api/exams/{exam_id} -> ExamDetail`、`get_session(request: Request) -> AsyncIterator[AsyncSession]`、`create_app(settings=None, *, session_factory=None)`。

- [ ] **Step 1: 写失败测试**

```python
async def test_list_and_get_exam(app_client, sample_exam):
    response = await app_client.get("/api/exams")
    assert response.status_code == 200
    assert response.json()[0]["question_count"] == 3

    response = await app_client.get(f"/api/exams/{sample_exam['exam_id']}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 3
    assert body["questions"][2]["question_type"] == "subjective"

    assert (await app_client.get("/api/exams/99999")).status_code == 404
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_exam_api.py -v`
Expected: FAIL，路由或 fixture 不存在。

- [ ] **Step 3: 写 Pydantic 响应模型**

`app/schemas/exam.py`：

```python
class ExamSummary(BaseModel):
    id: int
    exam_code: str
    year: int
    paper_name: str
    question_count: int

class PassageRead(BaseModel):
    id: int
    title: str | None
    author: str | None
    content: str
    paragraphs: list[str]

class QuestionRead(BaseModel):
    id: int
    question_number: int
    order_index: int
    stem: str
    question_type: str
    options: list[dict]
    correct_answer: str | None
    answer_points: list[str]
    explanation: str | None
    max_score: float | None
    active_rubric: dict | None = None

class ExamDetail(BaseModel):
    id: int
    exam_code: str
    year: int
    paper_name: str
    passage: PassageRead
    questions: list[QuestionRead]
```

- [ ] **Step 4: 写查询和路由**

`app/db/repositories.py`：

```python
async def list_exams(session: AsyncSession) -> list[ExamSummary]: ...
async def get_exam_detail(session: AsyncSession, exam_id: int) -> ExamDetail | None: ...
```

`list_exams` 用 `count(questions.id)` 聚合，按 `source_index` 排序。`get_exam_detail` 用 `selectinload` 加载 passage 和 questions；没有 passage 时返回 404。每道题再查 `rubrics.active=true` 的行，放进 `active_rubric`，没有就返回 `None`。

`app/api/deps.py`：

```python
async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session
```

`app/main.py` 的 `create_app` 增加 `session_factory` 参数；如果没传，用 settings 建 engine/factory，`lifespan` 结束时 dispose。注册 `exams` router。

- [ ] **Step 5: 更新 `tests/conftest.py` 的 `app_client`**

```python
@pytest.fixture
async def app(test_settings, test_session_factory):
    return create_app(test_settings, session_factory=test_session_factory)

@pytest.fixture
async def app_client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
```

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_exam_api.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/schemas app/db/repositories.py app/api app/main.py tests/conftest.py tests/integration/test_exam_api.py
git commit -m "feat: add exam read API"
```

---

### Task 4: 初始化脚本、错因分类和默认分值

**Files:**
- Create: `app/db/seed.py`
- Create: `scripts/init_db.py`
- Create: `scripts/init_scores.py`
- Modify: `tests/conftest.py`
- Test: `tests/integration/test_seed.py`

**Interfaces:**
- Consumes: 模型、`Settings`、`create_engine_and_factory`。
- Produces: `seed_error_categories(session) -> None`、`apply_default_scores(session, mcq: float = 3, subjective: float = 6) -> int`。

- [ ] **Step 1: 写失败测试**

```python
async def test_apply_default_scores(session, sample_exam):
    q1 = await session.get(Question, sample_exam["questions"][0]["id"])
    q3 = await session.get(Question, sample_exam["questions"][2]["id"])
    q1.max_score = None
    q3.max_score = None
    await session.commit()

    updated = await apply_default_scores(session)
    assert updated == 2
    assert (await session.get(Question, q1.id)).max_score == 3
    assert (await session.get(Question, q3.id)).max_score == 6
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_seed.py -v`
Expected: FAIL，`app.db.seed` 不存在。

- [ ] **Step 3: 写 `app/db/seed.py`**

`ERROR_CATEGORIES` 用设计文档 §6.6 的九个 code/name/description。`seed_error_categories` 用 `INSERT ... ON CONFLICT (code) DO NOTHING`。`apply_default_scores` 只更新 `max_score IS NULL` 的行，返回更新行数。

- [ ] **Step 4: 写脚本**

`scripts/init_db.py`：用 `docker exec -i agent_test-postgres psql -U postgres -d Mark -v ON_ERROR_STOP=1` 执行 `db/schema.sql`，再用 SQLAlchemy 调 `seed_error_categories`。支持 `--container`、`--database`、`--user`。

`scripts/init_scores.py`：读 `Settings`，建 engine/factory，调 `apply_default_scores`，打印更新行数。

- [ ] **Step 5: 更新测试 fixture**

`tests/conftest.py` 的清表步骤后调用 `seed_error_categories(session)`，保证每个测试都有错因分类。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_seed.py -v`
Expected: PASS。

- [ ] **Step 7: 手动跑一次脚本（可选）**

```powershell
.venv\Scripts\python.exe scripts\init_db.py
.venv\Scripts\python.exe scripts\init_scores.py
```

Expected: 没有异常；`init_scores.py` 打印 0 或题库里空分值的数量。

- [ ] **Step 8: 提交（仅当用户要求提交时）**

```powershell
git add app/db/seed.py scripts/init_db.py scripts/init_scores.py tests/conftest.py tests/integration/test_seed.py
git commit -m "feat: seed error categories and default scores"
```

---

### Task 5: FastAPI 托管静态复核页

**Files:**
- Modify: `app/main.py`
- Test: `tests/integration/test_static.py`

**Interfaces:**
- Consumes: `app/static/index.html`。
- Produces: `GET /` 返回静态页；`GET /docs` 仍然可用。

- [ ] **Step 1: 写失败测试**

```python
async def test_static_index(app_client):
    response = await app_client.get("/")
    assert response.status_code == 200
    assert "智能阅卷" in response.text

async def test_docs_still_available(app_client):
    assert (await app_client.get("/docs")).status_code == 200
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_static.py -v`
Expected: FAIL，`/` 404。

- [ ] **Step 3: 挂载静态目录**

在 `app/main.py` 注册完所有路由后：

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
```

必须放在最后，避免遮蔽 `/api` 和 `/docs`。

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_static.py -v`
Expected: PASS。

- [ ] **Step 5: 提交（仅当用户要求提交时）**

```powershell
git add app/main.py tests/integration/test_static.py
git commit -m "feat: serve static review page from FastAPI"
```

---

### Task 6: LLM 客户端、结构化输出和 JSON 修复

**Files:**
- Create: `app/schemas/grading.py`
- Create: `app/services/__init__.py`
- Create: `app/services/llm.py`
- Create: `app/services/observability.py`
- Create: `tests/fakes.py`
- Test: `tests/unit/test_llm_json.py`
- Test: `tests/unit/test_llm_client.py`

**Interfaces:**
- Consumes: `Settings`。
- Produces: `LLMOutputError`、`LLMResult[T](value: T, repaired: bool)`、`LLMProtocol`、`LLMClient`、`extract_json(text: str) -> str`、`build_openai_client(settings: Settings) -> AsyncOpenAI`、`FakeLLMClient(responses: dict[str, object])`；Pydantic 模型 `QuestionStructure`、`RubricPoint`、`RubricOutput`、`PointResult`、`GradeOutput`、`ErrorAnalysisOutput`、`ReportNarrative`。

- [ ] **Step 1: 写失败的 JSON 测试**

```python
def test_extract_json_plain():
    assert extract_json('{"value": 1}') == '{"value": 1}'

def test_extract_json_fenced():
    assert extract_json('```json\n{"value": 1}\n```') == '{"value": 1}'

def test_extract_json_with_prose():
    assert extract_json('结果如下：{"value": 1}，完毕') == '{"value": 1}'
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_llm_json.py -v`
Expected: FAIL，函数不存在。

- [ ] **Step 3: 写 Pydantic 模型**

`app/schemas/grading.py` 定义：

```python
class QuestionStructure(BaseModel):
    task_type: str
    answer_requirements: list[str] = []
    knowledge_points: list[str] = []
    scoring_mode: Literal["exact_match", "rubric"]
    notes: str = ""

class RubricPoint(BaseModel):
    id: str
    description: str
    score: float = Field(gt=0)
    evidence_hint: str = ""

class RubricOutput(BaseModel):
    max_score: float = Field(gt=0)
    points: list[RubricPoint]
    scoring_rules: str = ""

class PointResult(BaseModel):
    point_id: str
    hit: bool
    score: float = Field(ge=0)
    evidence: str = ""
    comment: str = ""

class GradeOutput(BaseModel):
    score: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    point_results: list[PointResult]
    rationale: str
    needs_review: bool = False

ERROR_TYPES = {
    "info_filter", "concept_misread", "incomplete_summary", "argument_analysis",
    "evidence_misuse", "expression", "off_topic", "blank", "mcq_wrong",
}

class ErrorAnalysisOutput(BaseModel):
    primary_error_type: Literal[...]  # 用九个 code 的字面量联合
    secondary_error_types: list[Literal[...]] = []
    evidence: str
    suggestion: str

class ReportNarrative(BaseModel):
    summary: str
    weaknesses: list[str]
    suggestions: list[str]
```

- [ ] **Step 4: 写 `LLMClient`**

`app/services/llm.py`：

```python
T = TypeVar("T", bound=BaseModel)

class LLMOutputError(Exception): ...

@dataclass
class LLMResult(Generic[T]):
    value: T
    repaired: bool

class LLMProtocol(Protocol):
    async def complete_json(self, prompt: str, schema: type[T], *, trace_name: str) -> LLMResult[T]: ...

class LLMClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        timeout: float = 60,
        max_retries: int = 2,
        concurrency: int = 5,
        completion_fn: Callable[[str], Awaitable[str]] | None = None,
    ) -> None: ...

    async def complete_json(self, prompt: str, schema: type[T], *, trace_name: str) -> LLMResult[T]: ...
```

`complete_json` 的规则：

- 在 prompt 末尾加“只输出 JSON，不要 markdown 代码块”。
- 用 `asyncio.Semaphore` 限制并发。
- 第一次调用后 `extract_json` + `schema.model_validate`。
- 失败时把错误和上一次输出拼进修复 prompt，再试一次（总调用次数 ≤ `max_retries + 1`）。
- 仍失败抛 `LLMOutputError`，异常带 `raw_text` 和 `errors`。
- 成功返回 `LLMResult(value=validated, repaired=...)`；第一次就成功 `repaired=False`，修复后成功 `repaired=True`。
- 默认 `completion_fn` 用 `client.chat.completions.create(..., response_format={"type": "json_object"}, temperature=0.1, timeout=timeout)`。

`build_openai_client`：`langfuse_enabled` 为真时返回 `langfuse.openai.AsyncOpenAI`，否则返回 `openai.AsyncOpenAI`。

`traceable(name, settings)` 在 `observability.py`：`langfuse_enabled=False` 时返回原函数，否则用 Langfuse 的 `observe`。

- [ ] **Step 5: 写重试测试**

```python
class StubCompletion:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
    async def __call__(self, prompt):
        self.calls += 1
        return self.outputs.pop(0)

async def test_complete_json_retries():
    stub = StubCompletion(['not json', '{"number": 1}'])
    client = LLMClient(AsyncOpenAI(api_key="test"), "test", completion_fn=stub)
    result = await client.complete_json("x", Answer, trace_name="test")
    assert result.value.number == 1
    assert result.repaired is True
    assert stub.calls == 2

async def test_complete_json_raises():
    stub = StubCompletion(["not json", "still not json", "bad"])
    client = LLMClient(AsyncOpenAI(api_key="test"), "test", max_retries=2, completion_fn=stub)
    with pytest.raises(LLMOutputError):
        await client.complete_json("x", Answer, trace_name="test")
```

`Answer` 是测试里定义的 `class Answer(BaseModel): number: int`。

- [ ] **Step 6: 写 `FakeLLMClient`**

`tests/fakes.py`：

```python
class FakeLLMClient:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[str] = []

    async def complete_json(self, prompt, schema, *, trace_name):
        self.calls.append(trace_name)
        value = self.responses[trace_name]
        if isinstance(value, Exception):
            raise value
        parsed = value if isinstance(value, schema) else schema.model_validate(value)
        return LLMResult(value=parsed, repaired=False)
```

- [ ] **Step 7: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_llm_json.py tests\unit\test_llm_client.py -v`
Expected: PASS。

- [ ] **Step 8: 提交（仅当用户要求提交时）**

```powershell
git add app/schemas/grading.py app/services tests/fakes.py tests/unit
git commit -m "feat: add structured LLM client with JSON repair"
```

---

### Task 7: 题目结构子图

**Files:**
- Create: `app/graph/__init__.py`
- Create: `app/graph/state.py`
- Create: `app/graph/subgraphs/__init__.py`
- Create: `app/graph/subgraphs/structure.py`
- Create: `app/prompts/structure.md`
- Test: `tests/unit/test_structure_subgraph.py`

**Interfaces:**
- Consumes: `LLMProtocol`、`QuestionStructure`。
- Produces: `QuestionContext`、`QuestionResult`、`ReviewItem`、`TeacherCorrection`、`GradingState`、`QuestionState`；`build_structure_subgraph(llm: LLMProtocol) -> CompiledStateGraph`；`fallback_structure(context: QuestionContext) -> QuestionStructure`。

- [ ] **Step 1: 写失败测试**

```python
async def test_structure_subgraph_valid(question_context):
    llm = FakeLLMClient({
        "structure": QuestionStructure(
            task_type="概括题",
            answer_requirements=["分点作答"],
            knowledge_points=["筛选并整合信息"],
            scoring_mode="rubric",
        )
    })
    graph = build_structure_subgraph(llm)
    result = await graph.ainvoke({"context": question_context})
    assert result["structure"]["task_type"] == "概括题"
    assert result["structure_fallback"] is False

async def test_structure_subgraph_falls_back(question_context):
    llm = FakeLLMClient({"structure": LLMOutputError("bad")})
    graph = build_structure_subgraph(llm)
    result = await graph.ainvoke({"context": question_context})
    assert result["structure_fallback"] is True
    assert result["structure"]["scoring_mode"] == "rubric"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_structure_subgraph.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写状态类型**

`app/graph/state.py` 定义设计文档 §6.1 的 `GradingState` 和 §6.2 的 `QuestionState`，并补上：

```python
class QuestionContext(TypedDict):
    question_id: int
    question_number: int
    exam_id: int
    question_type: Literal["single_choice", "subjective"]
    stem: str
    passage: str
    options: list[dict]
    correct_answer: str | None
    answer_points: list[str]
    explanation: str | None
    max_score: float
    student_answer: str | None
    selected_options: list[str]
    rubric: dict | None
    rubric_id: int | None

class QuestionResult(TypedDict):
    question_id: int
    question_number: int
    question_type: str
    ai_score: float
    max_score: float
    confidence: float
    needs_review: bool
    rationale: str
    point_results: list[dict]
    knowledge_points: list[str]
    error_type: str | None
    error_evidence: str | None
    final_score: float | None
    reviewed: bool

class ReviewItem(TypedDict):
    result_id: int
    question_id: int
    question_number: int
    question_type: str
    student_answer: str | None
    ai_score: float
    max_score: float
    confidence: float
    rationale: str
    point_results: list[dict]

class TeacherCorrection(TypedDict):
    result_id: int
    final_score: float
    instruction: str
    new_rationale: str | None

class GradingState(TypedDict, total=False):
    exam_id: int
    exam_code: str
    student_id: int
    student_name: str
    submission_id: int
    grading_run_id: int
    requested_rubric_version_id: int | None
    question_contexts: list[QuestionContext]
    results: Annotated[list[QuestionResult], operator.add]
    corrected_results: list[QuestionResult] | None
    review_items: list[ReviewItem]
    corrections: list[TeacherCorrection]
    report: dict | None
    total_score: float
    max_score: float
    error: str | None
```

`QuestionState` 字段：`context`、`structure`、`structure_fallback`、`rubric`、`rubric_fallback`、`result`。

- [ ] **Step 4: 写子图**

`build_structure_subgraph` 用 `StateGraph`：

```
START -> structure -> validate -> END
```

`structure` 节点调 `llm.complete_json(prompt, QuestionStructure, trace_name="structure")`，取 `response.value`；`validate` 检查 `task_type` 非空、`scoring_mode` 合法，失败就调 `fallback_structure` 并把 `structure_fallback=True`。`fallback_structure` 从 context 推导：选择题 `scoring_mode="exact_match"`，主观题 `scoring_mode="rubric"`，`task_type` 用“主观题”，其余为空数组。

`app/prompts/structure.md` 用中文，要求只输出 `QuestionStructure` 的 JSON。

- [ ] **Step 5: 加 `question_context` fixture**

在 `tests/conftest.py` 里从 `sample_exam` 的 Q3 构造 `QuestionContext`，包含 passage 文本、题干、参考答案、采分点、explanation、`max_score=6`、空学生答案。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_structure_subgraph.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/graph app/prompts/structure.md tests/conftest.py tests/unit/test_structure_subgraph.py
git commit -m "feat: add question structure subgraph"
```

---

### Task 8: 评分细则子图

**Files:**
- Create: `app/graph/subgraphs/rubric.py`
- Create: `app/prompts/rubric.md`
- Test: `tests/unit/test_rubric_subgraph.py`

**Interfaces:**
- Consumes: `LLMProtocol`、`RubricOutput`、`QuestionContext`。
- Produces: `RubricState`、`build_rubric_subgraph(llm, allow_fallback=True) -> CompiledStateGraph`、`validate_rubric(rubric: RubricOutput, max_score: float) -> list[str]`、`fallback_rubric(context: QuestionContext, max_score: float) -> RubricOutput`、`mcq_rubric(max_score: float) -> RubricOutput`。

- [ ] **Step 1: 写失败测试**

```python
def test_validate_rubric_rejects_wrong_sum():
    rubric = RubricOutput(
        max_score=6,
        points=[RubricPoint(id="p1", description="点1", score=2),
                RubricPoint(id="p2", description="点2", score=2)],
    )
    assert validate_rubric(rubric, 6) == ["细则分数之和 4.0 不等于 6.0"]

def test_fallback_rubric_sums_to_max(question_context):
    rubric = fallback_rubric(question_context, 6)
    assert rubric.max_score == 6
    assert sum(point.score for point in rubric.points) == 6
    assert len(rubric.points) == 3

def test_mcq_rubric():
    rubric = mcq_rubric(3)
    assert rubric.points[0].score == 3
    assert rubric.scoring_rules.startswith("答案与正确选项一致")
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_rubric_subgraph.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写校验和兜底**

`validate_rubric` 检查：points 非空、每个 `score > 0`、id 唯一、`abs(sum - max_score) < 1e-6`、`max_score` 与参数一致。返回错误字符串列表。

`fallback_rubric`：如果 `context["answer_points"]` 非空，按数量均分 `max_score`，步长 0.5，余数给最后一个点；如果为空，用一个“答出参考答案要点”的点占满分。

`mcq_rubric(max_score)`：一个点，id `correct`，描述“选出正确选项”，分数 `max_score`。

- [ ] **Step 4: 写子图**

`RubricState` 字段：`context`、`instruction`（校准用，普通批改为 None）、`rubric`、`rubric_fallback`、`rubric_errors`、`repair_attempts`。

图：

```
START -> draft -> validate -> END
                 validate -> repair -> validate
                 validate -> fallback -> END
```

`draft` 调 LLM 并取 `response.value`，`trace_name="rubric"`；`validate` 用 `validate_rubric`；失败且 `repair_attempts < 1` 时进 `repair`，否则进 `fallback`。`allow_fallback=False` 时 `fallback` 节点只设置 `rubric_errors`，不生成 rubric；校准服务据此报 422。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_rubric_subgraph.py -v`
Expected: PASS。

- [ ] **Step 6: 提交（仅当用户要求提交时）**

```powershell
git add app/graph/subgraphs/rubric.py app/prompts/rubric.md tests/unit/test_rubric_subgraph.py
git commit -m "feat: add rubric subgraph with validation and fallback"
```

---

### Task 9: 错因归类子图

**Files:**
- Create: `app/graph/subgraphs/error_analysis.py`
- Create: `app/prompts/error_analysis.md`
- Test: `tests/unit/test_error_analysis_subgraph.py`

**Interfaces:**
- Consumes: `LLMProtocol`、`ErrorAnalysisOutput`、`QuestionContext`、`QuestionResult`。
- Produces: `ErrorState`、`build_error_analysis_subgraph(llm) -> CompiledStateGraph`、`fallback_error_analysis(context, result) -> ErrorAnalysisOutput`。

- [ ] **Step 1: 写失败测试**

```python
async def test_error_analysis_valid(question_context, low_score_result):
    llm = FakeLLMClient({
        "error_analysis": ErrorAnalysisOutput(
            primary_error_type="incomplete_summary",
            evidence="只答出第一个采分点",
            suggestion="分点概括",
        )
    })
    graph = build_error_analysis_subgraph(llm)
    result = await graph.ainvoke({"context": question_context, "result": low_score_result})
    assert result["error_type"] == "incomplete_summary"

def test_blank_falls_back_to_blank(question_context, low_score_result):
    question_context["student_answer"] = ""
    output = fallback_error_analysis(question_context, low_score_result)
    assert output.primary_error_type == "blank"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_error_analysis_subgraph.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写子图和兜底**

`ErrorState` 字段：`context`、`result`、`error_type`、`error_evidence`、`suggestion`、`error_fallback`。

图：`START -> classify -> validate -> finalize -> END`。`classify` 调 LLM 并取 `response.value`，`trace_name="error_analysis"`；`validate` 检查 code 在 `ERROR_TYPES` 里；失败或 LLM 异常走 `fallback_error_analysis`。

兜底规则：答案为空 -> `blank`；选择题 -> `mcq_wrong`；其他 -> `incomplete_summary`。次要错因只保留合法 code。

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_error_analysis_subgraph.py -v`
Expected: PASS。

- [ ] **Step 5: 提交（仅当用户要求提交时）**

```powershell
git add app/graph/subgraphs/error_analysis.py app/prompts/error_analysis.md tests/unit/test_error_analysis_subgraph.py
git commit -m "feat: add error analysis subgraph"
```

---

### Task 10: 逐题批改子图（选择题短路 + 主观题完整流程）

**Files:**
- Create: `app/graph/subgraphs/grade_question.py`
- Create: `app/prompts/grade.md`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_question_subgraph.py`

**Interfaces:**
- Consumes: `build_structure_subgraph`、`build_rubric_subgraph`、`build_error_analysis_subgraph`、`LLMProtocol`、`Settings`。
- Produces: `build_question_subgraph(llm: LLMProtocol, settings: Settings) -> CompiledStateGraph`、`grade_mcq(context: QuestionContext, settings: Settings) -> QuestionResult`、`confidence_for(output: GradeOutput, context: QuestionContext, *, repaired: bool, rubric_fallback: bool) -> float`、`needs_review(result: QuestionResult, settings: Settings) -> bool`。

- [ ] **Step 1: 写失败测试**

```python
def test_mcq_correct(sample_exam, settings):
    result = grade_mcq(question_context(sample_exam, 0), settings)
    assert result["ai_score"] == 3
    assert result["confidence"] == 1
    assert result["needs_review"] is False

def test_mcq_invalid_selection_is_reviewed(sample_exam, settings):
    context = question_context(sample_exam, 0)
    context["selected_options"] = ["Z"]
    result = grade_mcq(context, settings)
    assert result["ai_score"] == 0
    assert result["needs_review"] is True

def test_confidence_rules(question_context):
    output = GradeOutput(score=0, confidence=0.9, point_results=[], rationale="x")
    context = question_context.copy()
    context["student_answer"] = "太短"
    assert confidence_for(output, context, repaired=True, rubric_fallback=False) < 0.7
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_question_subgraph.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写选择题和置信度**

`grade_mcq`：

- 合法选择：`selected_options` 恰好一个值，且在选项 label 集合里。
- 与 `correct_answer` 一致得 `max_score`，否则 0。
- `confidence=1.0`，`needs_review=False`，`point_results=[]`，`knowledge_points=[]`。
- 错选时 `error_type="mcq_wrong"`，evidence 写“学生选 X，正确答案 Y”。
- 非法选择：0 分，`needs_review=True`，`error_type="mcq_wrong"`，rationale “选择题答案无法识别”。

`confidence_for`：

- 取 `output.confidence`。
- `repaired` 减 0.1。
- `rubric_fallback` 减 0.2。
- 得分 0 或满分且 `len(student_answer or "") < 20` 减 0.15。
- 钳制到 `[0, 1]`。

- [ ] **Step 4: 写逐题子图**

`QuestionState` 用 Task 7 的定义。图：

```
START -> route
route -> grade_mcq (选择题)
route -> structure -> rubric -> grade -> route_after_grade
route_after_grade -> error_analysis (score < max_score)
route_after_grade -> finalize (score == max_score)
error_analysis -> finalize
grade_mcq -> finalize
finalize -> END
```

`structure`、`rubric`、`error_analysis` 用编译好的子图作为节点；`rubric` 子图接收 `context["rubric"]` 作为已有细则时直接跳过 LLM（在节点里判断）。`grade` 节点调 `llm.complete_json(prompt, GradeOutput, trace_name="grade")`，取 `response.value` 和 `response.repaired`；`repaired` 和 `state["rubric_fallback"]` 一起传给 `confidence_for`。

`grade` 校验：

- `abs(score - sum(point.score)) < 1e-6`。
- `0 <= score <= max_score`。
- `point_id` 都在当前细则里。
- 失败时 `LLMClient` 已经重试；仍失败就构造兜底结果：0 分、confidence 0、needs_review True、rationale “评分输出无法解析”。

`finalize` 汇总：`knowledge_points` 来自 structure，`error_type/error_evidence` 来自 error_analysis 或 MCQ，`final_score=None`、`reviewed=False`，最后用 `needs_review` 规则覆盖。

- [ ] **Step 5: 写主观题集成测试**

用 `FakeLLMClient` 提供 `structure`、`rubric`、`grade` 三组响应，跑完整子图：

- 正常答案：4/6，confidence 0.9，needs_review False。
- 低置信度：confidence 0.5，needs_review True。
- 空白答案：0 分，error_type blank，needs_review True。
- 满分：error_type 为 None，不调 `error_analysis`。
- 非法评分输出：point_id 不在细则里，或分数加总不等于 `score`，兜底为 0 分、confidence 0、needs_review True，不写非法结果。

检查 `llm.calls` 里没有 `error_analysis` 当满分时。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_question_subgraph.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/graph/subgraphs/grade_question.py app/prompts/grade.md tests/conftest.py tests/unit/test_question_subgraph.py
git commit -m "feat: add per-question grading subgraph"
```

---

### Task 11: 顶层图、并行分发和结果落库

**Files:**
- Create: `app/graph/nodes/__init__.py`
- Create: `app/graph/nodes/load_submission.py`
- Create: `app/graph/nodes/prepare_questions.py`
- Create: `app/graph/nodes/grade_question.py`
- Create: `app/graph/nodes/collect_results.py`
- Create: `app/graph/nodes/persist_results.py`
- Create: `app/graph/nodes/finalize.py`
- Create: `app/graph/builder.py`
- Create: `app/graph/checkpointer.py`
- Modify: `app/db/repositories.py`
- Test: `tests/integration/test_grading_graph.py`

**Interfaces:**
- Consumes: 所有子图、模型、`QuestionContext`、`QuestionResult`。
- Produces: `build_grading_graph(llm, session_factory, settings, checkpointer) -> CompiledStateGraph`；`create_checkpointer(settings) -> AsyncPostgresSaver`；仓库函数 `get_submission_bundle`、`get_active_rubric`、`save_grading_results`、`update_run_heartbeat`。

- [ ] **Step 1: 写失败测试**

```python
async def test_graph_grades_all_questions(test_session_factory, sample_submission, fake_llm, settings):
    graph = build_grading_graph(fake_llm, test_session_factory, settings, InMemorySaver())
    result = await graph.ainvoke(
        {"submission_id": sample_submission["submission_id"]},
        {"configurable": {"thread_id": "test-graph"}},
    )
    assert len(result["results"]) == 3
    assert result["results"][0]["ai_score"] == 3
    assert result["results"][2]["question_type"] == "subjective"

    async with test_session_factory() as session:
        rows = (await session.execute(select(GradingResult))).scalars().all()
    assert len(rows) == 3
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_grading_graph.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写仓库函数**

`app/db/repositories.py` 增加：

```python
async def get_submission_bundle(session, submission_id: int) -> dict | None: ...
async def get_active_rubric(session, question_id: int, requested_rubric_id: int | None = None) -> Rubric | None: ...
async def save_grading_results(session, run_id: int, results: list[QuestionResult]) -> list[ReviewItem]: ...
async def update_run_heartbeat(session, run_id: int) -> None: ...
```

`get_submission_bundle` 返回 submission、exam、passage、questions、answers、student 名称、exam_code。没有的 answer 按空答案处理。`save_grading_results` 用 `INSERT ... ON CONFLICT (run_id, question_id) DO UPDATE`，返回需要复核的 `ReviewItem` 列表（带 result_id）。未复核时 `final_score = ai_score`；`knowledge_points` 一起写入。

- [ ] **Step 4: 写数据装载节点**

`load_submission`：调 `get_submission_bundle`，把 submission、exam、passage、questions、answers、student 名称、exam_code 放进 state；调 `update_run_heartbeat`。

`prepare_questions`：把 bundle 转成 `question_contexts`；每道题查 active rubric（或 requested 覆盖），已有的放进 `context["rubric"]`（`{"max_score": rubric.total_score, "points": rubric.points, "scoring_rules": ""}`）和 `context["rubric_id"]`。

- [ ] **Step 5: 写批改和汇总节点**

`grade_question`：包装 `build_question_subgraph`，`ainvoke({"context": context})`，返回 `{"results": [result]}`。这样父图的 `results` reducer 自动追加。返回前调 `update_run_heartbeat`，逐题批改期间心跳不会停。

`collect_results`：按 `question_number` 排序，计算 `total_score`/`max_score` 到 state；不要返回 `results`，避免 reducer 重复。

- [ ] **Step 6: 写持久化和收尾节点**

`persist_results`：调 `save_grading_results`，返回 `review_items`；更新 heartbeat。

`finalize`：更新 `grading_runs.status='completed'`、`finished_at`、`submissions.status='completed'`、`total_score`、`max_score`、`graded_at`。

- [ ] **Step 7: 写图构建**

导入：`from langgraph.graph import START, END, StateGraph`；`from langgraph.types import Send`；测试用 `from langgraph.checkpoint.memory import InMemorySaver`；生产用 `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`。

`build_grading_graph`：

```python
graph = StateGraph(GradingState)
graph.add_node("load_submission", load_submission_node)
graph.add_node("prepare_questions", prepare_questions_node)
graph.add_node("grade_question", grade_question_node)
graph.add_node("collect_results", collect_results_node)
graph.add_node("persist_results", persist_results_node)
graph.add_node("finalize", finalize_node)
graph.add_edge(START, "load_submission")
graph.add_edge("load_submission", "prepare_questions")
graph.add_conditional_edges("prepare_questions", fan_out, ["grade_question"])
graph.add_edge("grade_question", "collect_results")
graph.add_edge("collect_results", "persist_results")
graph.add_edge("persist_results", "finalize")
graph.add_edge("finalize", END)
return graph.compile(checkpointer=checkpointer)
```

`fan_out` 返回 `[Send("grade_question", {"context": ctx}) for ctx in state["question_contexts"]]`。

`create_checkpointer`：

```python
def create_checkpointer(settings: Settings) -> AsyncPostgresSaver:
    dsn = settings.mark_database_url.replace("postgresql+psycopg://", "postgresql://")
    return AsyncPostgresSaver.from_conn_string(dsn)
```

`setup()` 在应用 lifespan 里 `await`；测试用 `InMemorySaver`，不需要 setup。如果安装的 LangGraph 版本只有 `MemorySaver` 这个名字，用它替代 `InMemorySaver`。`from_conn_string` 的具体生命周期按安装版本文档处理；如果它返回异步上下文管理器，就在 lifespan 里 `async with` 并保存 saver。

- [ ] **Step 8: 写假 LLM 完整响应**

在 `tests/conftest.py` 的 `fake_llm` fixture 里提供：

- `structure`：概括题。
- `rubric`：3 个点，各 2 分。
- `grade`：4 分，confidence 0.9，两个点命中。
- `error_analysis`：`incomplete_summary`。
- `student_report`、`class_report`：任意合法 narrative。

测试可以直接覆盖某一条响应来制造低置信度。

- [ ] **Step 9: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_grading_graph.py -v`
Expected: PASS。

- [ ] **Step 10: 提交（仅当用户要求提交时）**

```powershell
git add app/graph app/db/repositories.py tests/conftest.py tests/integration/test_grading_graph.py
git commit -m "feat: add top-level grading graph with fan-out and persistence"
```

---

### Task 12: 批改服务和答卷 API

**Files:**
- Create: `app/services/grading.py`
- Create: `app/schemas/submission.py`
- Create: `app/api/routers/submissions.py`
- Create: `app/api/routers/grading.py`
- Modify: `app/main.py`
- Modify: `app/api/deps.py`
- Test: `tests/integration/test_grading_api.py`

**Interfaces:**
- Consumes: `build_grading_graph`、`create_checkpointer`、`LLMClient`、模型。
- Produces: `GradingService(graph, session_factory, settings)`；`create_run(submission_id, requested_rubric_version_id=None) -> int`、`run(run_id) -> None`、`resume(run_id, corrections) -> None`、`get_run(run_id) -> GradingRunRead`；`mark_stale_runs(session_factory, older_than_minutes=30) -> int`；`POST /api/exams/{exam_id}/submissions`、`GET /api/submissions/{id}`、`POST /api/submissions/{id}/grade`、`GET /api/grading-runs/{id}`。

- [ ] **Step 1: 写失败测试**

```python
async def test_create_submission_and_grade(app_client, app, sample_exam, test_session_factory):
    response = await app_client.post(
        f"/api/exams/{sample_exam['exam_id']}/submissions",
        json={
            "student_id": sample_exam["student_id"],
            "answers": [
                {"question_id": sample_exam["questions"][0]["id"], "selected_options": ["C"]},
                {"question_id": sample_exam["questions"][2]["id"], "answer_text": "只答出第一点"},
            ],
        },
    )
    assert response.status_code == 201
    submission_id = response.json()["id"]

    response = await app_client.post(f"/api/submissions/{submission_id}/grade")
    assert response.status_code == 202
    run_id = response.json()["grading_run_id"]

    async with test_session_factory() as session:
        run = await session.get(GradingRun, run_id)
        assert run is not None
        service = GradingService(app.state.graph, test_session_factory, app.state.settings)
        await service.run(run_id)

    response = await app_client.get(f"/api/grading-runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["status"] in {"completed", "awaiting_review"}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_grading_api.py -v`
Expected: FAIL，路由不存在。

- [ ] **Step 3: 写 Pydantic schema**

`app/schemas/submission.py`：

```python
class AnswerCreate(BaseModel):
    question_id: int
    selected_options: list[str] = []
    answer_text: str | None = None

class SubmissionCreate(BaseModel):
    student_id: int
    answers: list[AnswerCreate]

class SubmissionRead(BaseModel):
    id: int
    exam_id: int
    student_id: int
    status: str
    total_score: float | None
    max_score: float | None
    answers: list[dict] = []

class GradingRunRead(BaseModel):
    id: int
    submission_id: int
    status: str
    model: str | None
    error: str | None
    results: list[dict]
    review_items: list[dict]
```

- [ ] **Step 4: 写服务**

`GradingService.run`：

- 读 run 和 submission；状态不是 pending/running 时直接返回。
- 设置 `running`、`started_at`、`heartbeat_at`。
- `config={"configurable": {"thread_id": run.graph_thread_id}}`。
- 初始 state：`exam_id`、`submission_id`、`grading_run_id`、`requested_rubric_version_id`、`results=[]`、`corrections=[]`。
- `result = await graph.ainvoke(state, config)`。
- 如果 `"__interrupt__" in result`，设 `awaiting_review`；否则 `completed`。
- 异常时设 `failed` 和 `error`。

`create_run`：插入 `grading_runs`，`graph_thread_id=uuid4().hex`，`model=settings.model_name`，`prompt_version=settings.prompt_version`，状态 pending。

`get_run`：查 run、results、review_items；submission 不存在或 run 不存在返回 None。

- [ ] **Step 5: 写路由和 app 组装**

- `POST /api/exams/{exam_id}/submissions`：校验 exam、student、question 归属；已存在 `(exam_id, student_id)` 返回 409；创建 submission 和 answers；返回 201。
- `GET /api/submissions/{id}`：返回 SubmissionRead（含 `student_answers` 的答案）和当前 run 摘要。
- `POST /api/submissions/{id}/grade`：`create_run`，`background_tasks.add_task(service.run, run_id)`，返回 202。
- `GET /api/grading-runs/{id}`：返回 GradingRunRead；404。
- `create_app` 增加 `llm`、`checkpointer` 参数；lifespan 里建 LLM、graph、checkpointer，存到 `app.state`。测试传 `FakeLLMClient` 和 `InMemorySaver`。
- 更新 `tests/conftest.py` 的 `app` fixture：`create_app(test_settings, session_factory=test_session_factory, llm=fake_llm, checkpointer=InMemorySaver())`。`fake_llm` 来自 Task 11 的 fixture。
- `app/services/grading.py` 增加 `mark_stale_runs(session_factory, older_than_minutes=30) -> int`：把 `status='running'` 且 `heartbeat_at < now() - interval '30 minutes'` 的 run 置为 `failed`，`error='stale heartbeat'`。`create_app` 启动时调用。测试插入一个心跳在 31 分钟前的 running run，断言它变成 failed。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_grading_api.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/services/grading.py app/schemas/submission.py app/api app/main.py tests/integration/test_grading_api.py
git commit -m "feat: add grading service and submission API"
```

---

### Task 13: 模拟班级和答卷

**Files:**
- Create: `app/services/synthetic.py`
- Create: `scripts/seed_synthetic.py`
- Create: `app/api/routers/dev.py`
- Modify: `app/main.py`
- Test: `tests/integration/test_synthetic.py`

**Interfaces:**
- Consumes: 模型、`sample_exam`（测试）、题库数据（脚本）。
- Produces: `seed_synthetic(session, exam_code: str, students: int = 10, class_name: str = "高二(1)班", seed: int = 42) -> dict`；`POST /api/dev/seed-synthetic`。

- [ ] **Step 1: 写失败测试**

```python
async def test_seed_synthetic_is_deterministic(session, sample_exam):
    first = await seed_synthetic(session, sample_exam["exam_code"], students=3, seed=7)
    assert first["student_count"] == 3
    assert first["submission_count"] == 3
    assert first["answer_count"] == 9

    profiles = (await session.execute(select(StudentAnswer.synthetic_profile))).scalars().all()
    assert set(profiles) <= {"full", "partial", "wrong", "blank"}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_synthetic.py -v`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写生成逻辑**

- 通过 `exam_code` 查 exam、questions。
- 创建 `ClassGroup` 和 `students` 个 `Student`，姓名 `模拟学生01` 起，`student_no=f"SYN-{uuid4().hex[:8]}-{i:02d}"`。脚本重复运行时不会撞唯一约束。
- 每个学生一份 `Submission(source='synthetic')`。
- 每道题一条 `StudentAnswer`：
  - 选择题：70% 正确选项，30% 随机错误选项；`selected_options=[label]`。
  - 主观题：full 30% 用 `correct_answer`；partial 40% 拼前一半 `answer_points`；wrong 20% 用固定离题句；blank 10% 用空字符串。
  - 用 `random.Random(seed)`，每题按 `question_number` 固定抽样顺序。
- 返回 `{"class_id", "student_count", "submission_count", "answer_count"}`。

- [ ] **Step 4: 写脚本和路由**

`scripts/seed_synthetic.py`：参数 `--exam-code`、`--students`、`--class-name`、`--seed`；建 engine/factory，调服务，打印结果。

`POST /api/dev/seed-synthetic`：请求 `{"exam_code", "students", "class_name", "seed"}`；调服务；返回计数。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_synthetic.py -v`
Expected: PASS。

- [ ] **Step 6: 提交（仅当用户要求提交时）**

```powershell
git add app/services/synthetic.py app/api/routers/dev.py scripts/seed_synthetic.py app/main.py tests/integration/test_synthetic.py
git commit -m "feat: add deterministic synthetic submissions"
```

---

### Task 14: 人工复核：interrupt、恢复和教师修正

**Files:**
- Create: `app/graph/nodes/review_gate.py`
- Create: `app/graph/nodes/human_review.py`
- Create: `app/graph/nodes/apply_corrections.py`
- Modify: `app/graph/builder.py`
- Modify: `app/services/grading.py`
- Create: `app/api/routers/review.py`
- Modify: `app/schemas/grading.py`
- Test: `tests/integration/test_review_flow.py`

**Interfaces:**
- Consumes: `GradingState`、`persist_results` 返回的 `review_items`、`GradingService.resume`。
- Produces: `POST /api/grading-runs/{run_id}/review`；`ReviewRequest`、`ReviewCorrection`；`GradingService.resume(run_id, corrections)`。

- [ ] **Step 1: 写失败测试**

```python
async def test_low_confidence_triggers_interrupt_and_resume(
    test_session_factory, sample_submission, settings
):
    llm = FakeLLMClient({
        "structure": QuestionStructure(task_type="概括题", scoring_mode="rubric"),
        "rubric": RubricOutput(max_score=6, points=[
            RubricPoint(id="p1", description="点1", score=2),
            RubricPoint(id="p2", description="点2", score=2),
            RubricPoint(id="p3", description="点3", score=2),
        ]),
        "grade": GradeOutput(
            score=4, confidence=0.5,
            point_results=[
                PointResult(point_id="p1", hit=True, score=2),
                PointResult(point_id="p2", hit=True, score=2),
                PointResult(point_id="p3", hit=False, score=0),
            ],
            rationale="低置信度",
        ),
        "error_analysis": ErrorAnalysisOutput(
            primary_error_type="incomplete_summary", evidence="缺一点", suggestion="分点"
        ),
        "student_report": ReportNarrative(summary="x", weaknesses=[], suggestions=[]),
    })
    graph = build_grading_graph(llm, test_session_factory, settings, InMemorySaver())
    config = {"configurable": {"thread_id": "review-test"}}
    first = await graph.ainvoke({"submission_id": sample_submission["submission_id"]}, config)
    assert "__interrupt__" in first

    items = first["__interrupt__"][0].value["items"]
    result_id = items[0]["result_id"]
    second = await graph.ainvoke(
        Command(resume={"corrections": [{"result_id": result_id, "final_score": 5, "instruction": "补一点"}]}),
        config,
    )
    assert second["corrected_results"][0]["final_score"] == 5
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_review_flow.py -v`
Expected: FAIL，复核节点不存在或没有 interrupt。

- [ ] **Step 3: 写三个节点**

- `review_gate`：如果 `state["review_items"]` 非空返回 `"human_review"`，否则 `"build_report"`。
- `human_review`：

```python
async def human_review_node(state):
    corrections = interrupt({"items": state["review_items"]})
    return {"corrections": corrections.get("corrections", [])}
```

- `apply_corrections`：校验每个 `result_id` 属于当前 run；更新 `grading_results.final_score`、`reviewed=True`，写 `teacher_corrections`（旧分、新分、教师原话、旧理由、当前细则快照）；重新计算 submission 总分；返回 `{"review_items": [], "corrected_results": [...]}`。不要返回 `results`，否则 `operator.add` reducer 会追加重复结果。

图新增（Task 16 之前先接到 `finalize`，Task 16 再把出口换成报告节点）：

```
persist_results -> review_gate
review_gate -> human_review (有复核)
review_gate -> finalize (无复核)
human_review -> apply_corrections -> finalize
```

- [ ] **Step 4: 写服务和路由**

`GradingService.resume(run_id, corrections)`：

- 读 run；状态不是 `awaiting_review` 抛 `ConflictError`。
- 调 `graph.ainvoke(Command(resume={"corrections": corrections}), config)`。
- 成功后设 `completed`；异常设 `failed`。
- 重复调用不会重复应用修正。

`POST /api/grading-runs/{run_id}/review`：

- 读 run；状态不是 `awaiting_review` 返回 409。
- 校验每个 `result_id` 属于该 run，否则 400。
- `background_tasks.add_task(service.resume, run_id, corrections)`。
- 返回 202。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_review_flow.py -v`
Expected: PASS。

- [ ] **Step 6: 补重复复核和重启恢复测试**

同一 run 第二次 POST 返回 409；`teacher_corrections` 只有一条。

另加一个 `AsyncPostgresSaver` 测试：用 `Mark_test` 建 checkpointer，`await checkpointer.setup()`，跑一次到 interrupt；新建 `build_grading_graph` 实例，用同一个 checkpointer 和 `thread_id` 恢复，确认能完成。这是“服务重启后恢复”的验证，测试结束用唯一的 `thread_id` 避免互相污染。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/graph app/services/grading.py app/api/routers/review.py app/schemas/grading.py app/graph/builder.py tests/integration/test_review_flow.py
git commit -m "feat: add human review interrupt and resume flow"
```

---

### Task 15: 自然语言校准和全量重批

**Files:**
- Create: `app/services/calibration.py`
- Create: `app/schemas/rubric.py`
- Create: `app/api/routers/calibration.py`
- Create: `app/api/routers/rubrics.py`
- Modify: `app/main.py`
- Modify: `app/db/repositories.py`
- Test: `tests/integration/test_calibration.py`

**Interfaces:**
- Consumes: `build_rubric_subgraph`、`GradingService.create_run`、模型。
- Produces: `calibrate(session, llm, exam_id, question_id, instruction, re_grade_all) -> dict`；`POST /api/exams/{exam_id}/calibrate`；`CalibrationRequest`、`CalibrationRead`。

- [ ] **Step 1: 写失败测试**

```python
async def test_calibration_creates_new_active_rubric(app_client, sample_exam, session):
    old = await get_active_rubric(session, sample_exam["questions"][2]["id"])
    response = await app_client.post(
        f"/api/exams/{sample_exam['exam_id']}/calibrate",
        json={"question_id": sample_exam["questions"][2]["id"],
              "instruction": "第三点只要提到倾向性标题就给分",
              "re_grade_all": False},
    )
    assert response.status_code == 200
    new_id = response.json()["rubric_id"]
    assert new_id != old.id
    await session.expire_all()
    assert (await session.get(Rubric, old.id)).active is False
    assert (await session.get(Rubric, new_id)).active is True
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_calibration.py -v`
Expected: FAIL，路由不存在。

- [ ] **Step 3: 写校准服务**

- 校验 exam/question 存在，question 属于 exam。
- 读 active rubric。
- 调 `build_rubric_subgraph(llm, allow_fallback=False)`，state 带 `context` 和 `instruction`。
- 如果 `rubric_fallback` 或没有 `rubric`，抛 `CalibrationError`，路由返回 422。
- 新版本号 = 当前最大 version + 1；旧 rubric `active=False`；写新 rubric `source='teacher'`、`instruction=...`；写 `calibration_logs`。
- `re_grade_all=True` 时，查该 exam 的 submissions，逐份 `GradingService.create_run(..., requested_rubric_version_id=new.id)`，返回 run IDs；路由用 background tasks 启动这些 run。

- [ ] **Step 4: 写路由和 schema**

`CalibrationRequest`：`question_id`、`instruction`、`re_grade_all=False`。`CalibrationRead`：`rubric_id`、`version`、`points`、`total_score`、`rerun_ids`。`RubricRead`：`id`、`question_id`、`version`、`points`、`total_score`、`source`、`active`、`created_at`。

`POST /api/exams/{exam_id}/calibrate` 返回 200；校准失败 422；exam/question 不存在 404。

`GET /api/questions/{question_id}/rubrics` 返回该题所有细则版本，按 `version DESC` 排序；question 不存在返回 404，存在但没有细则返回空列表。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_calibration.py -v`
Expected: PASS。

- [ ] **Step 6: 提交（仅当用户要求提交时）**

```powershell
git add app/services/calibration.py app/schemas/rubric.py app/api/routers/calibration.py app/main.py app/db/repositories.py tests/integration/test_calibration.py
git commit -m "feat: add natural language rubric calibration"
```

---

### Task 16: 学生报告和班级报告

**Files:**
- Create: `app/services/reports.py`
- Create: `app/schemas/report.py`
- Create: `app/api/routers/reports.py`
- Create: `app/prompts/student_report.md`
- Create: `app/prompts/class_report.md`
- Create: `app/graph/nodes/build_report.py`
- Create: `app/graph/nodes/persist_report.py`
- Modify: `app/graph/builder.py`
- Modify: `tests/conftest.py`
- Test: `tests/integration/test_reports.py`

**Interfaces:**
- Consumes: 模型、`LLMProtocol`、`ReportNarrative`。
- Produces: `build_class_aggregate(session, class_id, exam_id) -> dict`、`generate_student_report(llm, student_name, exam_code, results, contexts) -> dict`、`generate_class_report(llm, class_name, exam_code, aggregate) -> dict`；`GET /api/students/{id}/report?exam_id=`、`GET /api/classes/{id}/report?exam_id=`；`ReportRead`。

- [ ] **Step 1: 写失败测试**

```python
async def test_class_report_uses_latest_completed_run(app_client, seeded_two_runs_same_submission):
    response = await app_client.get(
        f"/api/classes/{seeded_two_runs_same_submission['class_id']}/report",
        params={"exam_id": seeded_two_runs_same_submission["exam_id"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["submission_count"] == 1
    assert body["data"]["average_score"] == 5.0  # 旧 run 3 分，新 run 5 分
    assert "平均分" in body["content_markdown"]
```

测试 fixture `seeded_two_runs_same_submission` 给同一份 submission 建两个 `completed` run：旧 run 总分 3，新 run 总分 5；报告必须用 5。

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_reports.py -v`
Expected: FAIL，路由不存在。

- [ ] **Step 3: 写聚合**

`build_class_aggregate`：

- 对每个 submission 取最新 `completed` 的 run：用 `DISTINCT ON (submission_id) ... ORDER BY created_at DESC` 或窗口函数。
- 只统计当前 run 的 `grading_results`。
- 聚合：submission_count、average_score、max_score、score_distribution（按百分比分段）、per_question（题号、平均得分率）、error_type_counts、knowledge_point_counts、weak_points（按出现次数排序取前 5）。
- 返回纯 JSON，不包含学生姓名。

- [ ] **Step 4: 写报告生成**

`generate_student_report` 和 `generate_class_report` 调 `llm.complete_json(..., ReportNarrative, trace_name="student_report"/"class_report")`，取 `response.value`。把 `ReportNarrative` 转成 markdown：

```markdown
## 总体
{summary}

## 薄弱点
- ...

## 建议
- ...
```

保存 `reports` 行：`scope_type`、`scope_id`、`exam_id`、`content_markdown`、`data`。`ReportRead` 就是这五个字段。

- [ ] **Step 5: 接图节点和路由**

- `build_report` 节点用 `state.get("corrected_results") or state["results"]` 调 `generate_student_report`，把结果放 `state["report"]`。
- `persist_report` 写 `reports`。
- 图的边：`review_gate -> build_report`（无复核）、`apply_corrections -> build_report`（复核后）、`build_report -> persist_report -> finalize`。
- `GET /api/students/{id}/report`：先找该学生该考试最新的 `reports`；没有就找最新 completed run，重建 results/contexts，生成并保存。
- `GET /api/classes/{id}/report`：聚合 + 生成 + 保存。

- [ ] **Step 6: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_reports.py -v`
Expected: PASS。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add app/services/reports.py app/schemas/report.py app/api/routers/reports.py app/prompts app/graph tests/integration/test_reports.py
git commit -m "feat: add student and class reports"
```

---

### Task 17: 坏案例接口

**Files:**
- Modify: `app/api/routers/dev.py`
- Create: `app/services/bad_cases.py`
- Test: `tests/integration/test_bad_cases.py`

**Interfaces:**
- Consumes: `grading_results`、`teacher_corrections`、最新 completed run。
- Produces: `GET /api/dev/bad-cases?exam_id=`；`BadCaseRead`。

- [ ] **Step 1: 写失败测试**

```python
async def test_bad_cases_returns_low_confidence_and_override(app_client, seeded_bad_cases):
    response = await app_client.get("/api/dev/bad-cases", params={"exam_id": seeded_bad_cases["exam_id"]})
    assert response.status_code == 200
    body = response.json()
    assert any(item["reason"] == "low_confidence" for item in body)
    assert any(item["reason"] == "teacher_override" for item in body)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_bad_cases.py -v`
Expected: FAIL，路由不存在。

- [ ] **Step 3: 写查询**

`bad_cases.py`：

- 取每个 submission 最新 completed run。
- `low_confidence`：`confidence < 0.6`。
- `teacher_override`：存在 `teacher_corrections` 且 `abs(new_score - old_score) >= 2`。
- `grading_failed`：`error_type='grading_failed'`。
- 返回题目、学生答案、AI 分数、最终分数、理由、错因、置信度。

`BadCaseRead` 字段：`question_id`、`question_number`、`student_name`、`student_answer`、`ai_score`、`final_score`、`confidence`、`rationale`、`error_type`、`reason`。

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_bad_cases.py -v`
Expected: PASS。

- [ ] **Step 5: 提交（仅当用户要求提交时）**

```powershell
git add app/services/bad_cases.py app/api/routers/dev.py tests/integration/test_bad_cases.py
git commit -m "feat: add bad case analysis endpoint"
```

---

### Task 18: Langfuse 接入

**Files:**
- Modify: `app/services/observability.py`
- Modify: `app/services/llm.py`
- Modify: `app/services/grading.py`
- Test: `tests/unit/test_observability.py`

**Interfaces:**
- Consumes: `Settings`。
- Produces: `traceable(name, settings)`、`trace_score(name, value, settings)`、`build_openai_client`；trace 名称和 score 名称按设计文档 §11.1。

- [ ] **Step 1: 写失败测试**

```python
def test_traceable_is_noop_when_disabled():
    settings = Settings(_env_file=None, langfuse_enabled=False)
    def fn():
        return 1
    assert traceable("x", settings)(fn) is fn

def test_build_openai_client_without_langfuse():
    settings = Settings(_env_file=None, langfuse_enabled=False, api_key="test")
    client = build_openai_client(settings)
    assert client.__class__.__module__.startswith("openai")
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_observability.py -v`
Expected: FAIL，函数不存在或签名不对。

- [ ] **Step 3: 实现**

- `traceable(name, settings)`：关闭时返回恒等装饰器；开启时用 Langfuse SDK 的 `observe`。具体 API 以安装版本为准，版本记录在 `pyproject.toml`。
- `trace_score(name, value, settings)`：关闭时 no-op；开启时用当前 trace 打分。
- `build_openai_client`：开启时 `from langfuse.openai import AsyncOpenAI`，关闭时 `from openai import AsyncOpenAI`。
- `GradingService.run` 里记录：`task_completed`、`needs_review`、`teacher_override`；LLM 客户端每次 JSON 校验结果记录 `json_valid`。

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_observability.py -v`
Expected: PASS。

- [ ] **Step 5: 手动验证（需要真实 key）**

启动服务，批改一份样本，打开 `http://localhost:3000`，确认能看到 `grade_submission` trace、子图 span 和 DeepSeek generation。没有 key 或不想产生费用时跳过，记录在 README。

- [ ] **Step 6: 提交（仅当用户要求提交时）**

```powershell
git add app/services/observability.py app/services/llm.py app/services/grading.py tests/unit/test_observability.py
git commit -m "feat: add Langfuse tracing and scores"
```

---

### Task 19: 评测脚本和黄金样本

**Files:**
- Create: `scripts/evaluate.py`
- Create: `tests/fixtures/golden_grading.jsonl`
- Test: `tests/unit/test_evaluate_metrics.py`

**Interfaces:**
- Consumes: `GradingService`/`GradingGraph`、`FakeLLMClient`（测试）、真实 LLM（手动）。
- Produces: `compute_metrics(cases: list[dict], predictions: list[dict]) -> dict`。

- [ ] **Step 1: 写失败测试**

```python
def test_compute_metrics():
    cases = [
        {"expected_score": 4, "expected_error_type": "incomplete_summary"},
        {"expected_score": 6, "expected_error_type": None},
    ]
    predictions = [
        {"score": 5, "error_type": "incomplete_summary", "json_valid": True, "confidence": 0.8},
        {"score": 6, "error_type": None, "json_valid": True, "confidence": 0.9},
    ]
    metrics = compute_metrics(cases, predictions)
    assert metrics["mae"] == 0.5
    assert metrics["exact_match_rate"] == 0.5
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["error_accuracy"] == 1.0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_evaluate_metrics.py -v`
Expected: FAIL，函数不存在。

- [ ] **Step 3: 写指标和脚本**

`compute_metrics` 返回 `mae`、`exact_match_rate`、`json_valid_rate`、`error_accuracy`、`avg_confidence`、`review_rate`。

`scripts/evaluate.py`：

- 读 `tests/fixtures/golden_grading.jsonl`。
- 每条含 `question_id`、`student_answer`、`expected_score`、`expected_error_type`。
- 用当前活跃细则跑评分；`--fake` 用固定预测，便于离线验证脚本。
- 打印指标；可选写 Langfuse `score_mae`、`faithfulness`。

- [ ] **Step 4: 写黄金样本**

至少 10 条，覆盖 3 道选择题、4 道主观题，满分/部分/错答/空白。样本从 `sample_exam` 或 `2023-新高考Ⅰ卷` 抄题干和参考答案，人工标期望分。不要把这些样本写进生产库。

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_evaluate_metrics.py -v`
Expected: PASS。

- [ ] **Step 6: 运行离线评测**

Run: `.venv\Scripts\python.exe scripts\evaluate.py --fake`
Expected: 打印六项指标，不报错。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add scripts/evaluate.py tests/fixtures/golden_grading.jsonl tests/unit/test_evaluate_metrics.py
git commit -m "feat: add offline evaluation metrics and golden cases"
```

---

### Task 20: 文档、全量验证和收尾

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `.env.example`
- Test: 无新增单元测试；跑全量验证。

**Interfaces:**
- Consumes: 全部。
- Produces: 可复现的 README、更新后的 CLAUDE.md。

- [ ] **Step 1: 重写 README**

去掉 GitLab 模板，写：

- 项目简介和目标流程。
- 当前进度：题库、静态页、后端。
- 环境要求：Python 3.12、Docker Postgres、Langfuse、DeepSeek。
- 初始化：`ensurepip`、`pip install -e ".[dev]"`、`init_db.py`、`init_scores.py`、`import_dataset.py --apply`、`seed_synthetic.py`。
- 启动：`scripts\run_server.py --port 8000`（Windows 上不要直接用 `python -m uvicorn`，psycopg async 需要 SelectorEventLoop）。
- API 流程：创建答卷 → 启动批改 → 查询 → 复核 → 校准 → 报告。
- 静态页：`http://localhost:8000/`。
- 测试和评测：`pytest -q`、`evaluate.py`。
- 已知限制：无 OCR、无 LoRA/vLLM、无登录、后台任务不持久、模拟数据不代表真实准确率。

- [ ] **Step 2: 更新 CLAUDE.md**

补充后端目录、常用命令、数据库 schema、测试库 `Mark_test`、设计文档和计划路径。所有内容用中文。

- [ ] **Step 3: 全量测试**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS。

- [ ] **Step 4: Lint**

Run: `.venv\Scripts\python.exe -m ruff check .`
Expected: 无错误。

- [ ] **Step 5: 启动服务并冒烟**

```powershell
.venv\Scripts\python.exe scripts\run_server.py --port 8000
```

另开终端：

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/exams
Invoke-WebRequest http://localhost:8000/ | Select-Object StatusCode
Invoke-WebRequest http://localhost:8000/docs | Select-Object StatusCode
```

Expected: `{"status": "ok"}`；21 个试卷组；静态页和 `/docs` 200。

- [ ] **Step 6: 可选真实模型冒烟**

用 `.env` 里的 DeepSeek key，挑一份模拟答卷跑一次真实批改。确认结果落库、Langfuse 有 trace。产生少量 API 费用，用户不同意就跳过。

- [ ] **Step 7: 提交（仅当用户要求提交时）**

```powershell
git add README.md CLAUDE.md .env.example
git commit -m "docs: add setup, API flow and limitations"
```

---

## Self-Review

- **Spec coverage:** 设计文档 §2.1 的每一项都有对应任务：题库/模型（Task 2）、试卷 API（Task 3）、默认分值（Task 4）、静态页（Task 5）、LLM 与 JSON（Task 6）、结构/细则/错因/逐题（Task 7–10）、顶层图（Task 11）、批改 API（Task 12）、模拟数据（Task 13）、复核（Task 14）、校准（Task 15）、报告（Task 16）、坏案例（Task 17）、Langfuse（Task 18）、评测（Task 19）、文档（Task 20）。§2.2 的非目标没有被任务实现。
- **Review Focus:** 五类风险分别落在 Task 10（空白、非法选择、非法评分输出）、Task 14（重复复核）、Task 16（最新 completed run）。
- **Type consistency:** `QuestionContext`、`QuestionResult`、`GradingState`、`GradingService` 的签名在各任务中一致；`knowledge_points` 在状态、模型、报告里统一出现。
- **Proportion:** 计划按可测试交付物拆分，测试代码只保留关键断言；DDL 和 prompt 细节引用设计文档，不重复整段实现。
