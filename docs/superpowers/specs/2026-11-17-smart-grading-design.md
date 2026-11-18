# 智能阅卷与学情分析系统：设计文档

- 日期：2026-11-17
- 状态：待评审
- 数据库：本地 PostgreSQL `Mark`
- 模型：DeepSeek `deepseek-chat`（OpenAI 兼容接口）
- 观测：本地 Langfuse（`http://localhost:3000`）

## 1. 背景与目标

从试卷和参考答案出发，系统先让 AI 逐题批改，教师只复核 AI 没把握的部分，最后产出班级和个人的学情报告。

项目职责里的五块是：LangGraph 多节点与子图、人机协同与 Prompt 校准、LoRA 微调与 vLLM 推理、错因分析与报告生成、Langfuse 质量监控。本期先做第 1、2、4、5 块中能在本机跑通的部分。第 3 块（LoRA/vLLM）本期明确不做，本机没有 GPU。

项目说明里的 95.6% 一致率、97.1% 免改率、2.8% 人工调整比例是目标，不是本期验收指标。本期没有真实教师复核数据，只能用模拟答卷验证链路和格式，不能对外声称达到了这三个数字。

本期要交付的是一个可运行的后端和一个静态教师复核页：

- FastAPI 提供提交答卷、启动批改、查询结果、教师复核、自然语言校准、学情报告接口。
- LangGraph 顶层图加三个子图，Postgres 做 checkpointer，教师复核用 `interrupt` 暂停和恢复。
- DeepSeek 负责主观题评分、评分细则生成、错因归类、报告生成。
- 选择题精确比对，不走 LLM 评分。
- 用题库生成模拟班级、学生和答卷。
- 所有 LLM 调用接 Langfuse，能看 trace、耗时、token 和自定义分数。
- 静态教师复核页 `app/static/index.html`，单文件 HTML/CSS/JS，用题库里的 2023 新高考Ⅰ卷做演示。页面不调用后端，接口接上后再把 mock 数据换成 fetch。
- 接口文档用 FastAPI 自带的 `/docs`。

## 2. 范围

### 2.1 本期做

- 扩展数据库：班级、学生、答卷、学生答案、评分细则、批改运行、批改结果、教师修正、校准记录、报告、错因分类。
- 填充题库的默认分值：选择题 3 分，主观题 6 分。原始数据没有分值，这是本期的假设，教师可以在校准流程里改。
- 模拟数据脚本：一个班、10 个学生、指定试卷的答卷，覆盖满分、部分得分、错答、空白四种情况。
- LangGraph 顶层图和三个子图：题目结构、评分细则、错因归类。
- 逐题并行批改：选择题精确比对；主观题按评分细则调用 LLM。
- 人工复核：低置信度、边界分数、JSON 校验失败、空白答案进入复核队列；教师提交修正后恢复图。
- 自然语言校准：教师用一句话改评分标准，生成新版本细则，可选对整场考试重新批改。
- 学情报告：学生个人报告、班级报告，结构化数据加 LLM 叙述。
- Langfuse：每次批改一条 trace，节点和 LLM 调用有 span，记录 JSON 有效率、任务完成、是否需要复核、教师是否改分。
- 评测脚本：一组人工标注的黄金样本，算分数误差、JSON 有效率、错因分类准确率。
- pytest 单元测试和集成测试，测试不调用真实 API。
- 静态教师复核页：班级 KPI、阅读材料、题目列表、AI 评分、采分点、复核队列、错因分布。单文件，无构建步骤，数据先用题库里的 2023 新高考Ⅰ卷。

### 2.2 本期不做

- 完整的单页应用框架（React/Vue）、登录、权限、多租户。本期只有一个静态演示页。
- 试卷扫描、图片 OCR、视觉模型、手写识别。
- 真实学生数据和真实教师账号。
- LoRA 微调、vLLM 部署、训练脚本。
- 生产级任务队列（Celery、RQ、Arq）。
- 移动端、消息通知、对象存储。
- 组卷和同类题生成。

### 2.3 反目标

- 不把关键词匹配包装成 AI 批改。主观题必须给出采分点命中情况、理由和证据。
- 不让未经校验的 LLM JSON 写进数据库。
- 不要求 GPU 才能跑演示。
- 不修改 `data/` 下的原始 docx。
- 不把 `.env`、API key、数据库密码写进代码或提交记录。
- 不为了凑出 95.6% 的指标去调模拟数据。

## 3. 约束

- Python 3.12，虚拟环境 `.venv`，目前没有第三方包和 pip。实现时先用 `python -m ensurepip --upgrade` 装 pip，再把依赖写进 `pyproject.toml` 后 `pip install -e .`。
- 数据库是 Docker 容器 `agent_test-postgres` 里的 `Mark` 库，PostgreSQL 18 + pgvector，用户名 `postgres`。本地连接串用环境变量 `MARK_DATABASE_URL`，默认 `postgresql+psycopg://postgres:postgres@localhost:5432/Mark`，不写进代码。
- DeepSeek 配置从 `.env` 读取：`API_KEY`、`BASE_URL=https://api.deepseek.com`、`MODEL_NAME=deepseek-chat`。这些变量名是现有 `.env` 的命名，应用层沿用。
- Langfuse 配置从 `.env` 读取：`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST=http://localhost:3000`、`LANGFUSE_ENABLED=true`。
- 题库已有 21 个试卷组、69 道题：58 道四选一、11 道主观题。答案和解析齐全，`max_score` 为空。
- 中文内容和中文 prompt。DeepSeek 的上下文长度足够放一篇阅读材料加一道题，不需要分段。
- 本机没有 NVIDIA GPU，不能跑 LoRA 和 vLLM。

## 4. 方案选择

考虑过三种编排方式：

**A. LangGraph 原生图 + Postgres checkpointer + `interrupt`。** 顶层图负责装载、并行分发、汇总；三个子图分别做结构、细则、错因；教师复核用 `interrupt` 暂停，`Command(resume=...)` 恢复。状态由 checkpointer 持久化。最贴合“多节点 + 子图 + State 管理”的要求。代价是状态设计、checkpoint 表和恢复逻辑更复杂。

**B. LangGraph 线性流水线，复核状态放业务表。** 图只做批改，复核后新开一次运行。实现简单，但人机协同不在图里，和项目职责的描述对不上。

**C. 不用 LangGraph，普通异步服务串 LLM 调用。** 最快，但去掉核心卖点。

选 A。教师复核、自然语言校准、报告生成都在同一套图里完成，后面加节点不用换框架。

其他决定：

- 逐题并行，不把整张卷子塞进一个 prompt。每道题独立上下文，失败可以单独重试，评分理由也能落到题级。
- 选择题精确比对；主观题 LLM 按评分细则打分。
- 用模拟答卷，不用真实学生数据。
- 首期不做 LoRA/vLLM。LLM 调用封装成客户端，后面换 vLLM 只改配置。
- 不做网页，接口和 OpenAPI 文档就是交付面。

## 5. 总体架构

```
客户端 / curl / OpenAPI
        |
        v
FastAPI 路由层
        |
        v
服务层：grading / calibration / reports / synthetic / llm / observability
        |
        +--> LangGraph 运行时（AsyncPostgresSaver）
        |       |
        |       +--> 顶层图：装载 -> 分发 -> 汇总 -> 复核 -> 报告
        |       +--> 子图：structure / rubric / error_analysis
        |
        +--> PostgreSQL Mark
        |       +--> 业务表：exams / questions / submissions / grading_results ...
        |       +--> checkpoint 表（独立 schema，建议 `graph`）
        |
        +--> DeepSeek API
        |
        +--> Langfuse
```

分层职责：

- 路由层只做参数校验、权限占位、调用服务、返回 Pydantic 响应。
- 服务层负责事务、图调用、状态恢复、报告和模拟数据。
- 图节点不直接拼 SQL，通过仓储函数读写业务表。
- LLM 客户端统一处理超时、重试、JSON 解析、Langfuse 包装。
- 观测层统一创建 trace、span、score，业务代码不散落 Langfuse 调用。

## 6. LangGraph 设计

### 6.1 顶层图

状态：

```python
class GradingState(TypedDict, total=False):
    exam_id: int
    submission_id: int
    grading_run_id: int
    requested_rubric_version_id: int | None
    question_contexts: list[QuestionContext]
    results: Annotated[list[QuestionResult], operator.add]
    review_items: list[ReviewItem]
    corrections: list[TeacherCorrection]
    report: dict | None
    error: str | None
```

节点和边：

```
START
  -> load_submission          读取试卷、材料、题目、学生答案、当前细则
  -> prepare_questions        组装每道题的上下文
  -> fan_out                  Send 并行分发到 grade_question
  -> grade_question           子图，逐题结构 -> 细则 -> 评分 -> 错因
  -> collect_results          按题号排序，算总分
  -> persist_results          写 grading_results，更新 run 状态
  -> review_gate              条件边：有复核项 -> human_review；没有 -> build_report
  -> human_review             interrupt，等待教师修正
  -> apply_corrections        写 teacher_corrections，更新最终分
  -> build_report             生成学生报告
  -> persist_report           写 reports
  -> finalize                 更新 grading_runs.status = completed
  -> END
```

失败处理：每个节点捕获异常，写入 `grading_runs.status='failed'` 和 `error`。单题失败不拖垮整卷：该题结果标记 `needs_review=true`、`confidence=0`、`error_type='grading_failed'`，继续处理其他题。整卷级失败才把 run 标为 failed。

### 6.2 逐题子图

```python
class QuestionState(TypedDict, total=False):
    context: QuestionContext
    structure: QuestionStructure | None
    rubric: Rubric | None
    result: QuestionResult | None
```

流程：

```
START
  -> structure_subgraph        题目结构识别
  -> rubric_subgraph           评分细则生成
  -> grade                     评分
  -> error_analysis_subgraph   错因归类（满分时跳过）
  -> END
```

选择题走一条短路：

- 结构：`task_type='选择判断'`，`scoring_mode='exact_match'`，不做 LLM 调用。
- 细则：正确选项得满分，其他 0 分，`total_score=3`。
- 评分：学生选项与 `questions.correct_answer` 完全一致得满分，否则 0 分，`confidence=1.0`。
- 错因：答错时固定为 `mcq_wrong`，证据写明学生选了什么、正确答案是什么。

主观题走完整子图。

### 6.3 structure 子图

目的：把题干、参考答案、解析变成结构化的答题要求，供细则和评分使用。

输入：题干、选项、参考答案、解析、题型。

输出 JSON：

```json
{
  "task_type": "概括题",
  "answer_requirements": ["概括原因", "分点作答"],
  "knowledge_points": ["筛选并整合信息", "理解重要概念"],
  "scoring_mode": "rubric",
  "notes": "参考答案有三个采分点"
}
```

校验：`task_type` 非空，`answer_requirements` 和 `knowledge_points` 是字符串数组，`scoring_mode` 只能是 `exact_match` 或 `rubric`。校验失败重试一次；再失败就用规则兜底：`task_type` 取题型，`answer_requirements=[]`，`knowledge_points=[]`，置信度降低并标记复核。

LLM 只对主观题调用。选择题用确定性结构，省调用。

### 6.4 rubric 子图

目的：把参考答案、解析、结构信息变成可打分的采分点。

输入：题目上下文、structure。

输出 JSON：

```json
{
  "max_score": 6,
  "points": [
    {"id": "p1", "description": "答出《独立报》用片面事实和数据", "score": 2, "evidence_hint": "《独立报》称藜麦消费量下降34%"},
    {"id": "p2", "description": "答出《纽约时报》援引研究", "score": 2, "evidence_hint": "儿童营养不良率上升"},
    {"id": "p3", "description": "答出《卫报》《独立报》用倾向性标题", "score": 2, "evidence_hint": "煽动性标题"}
  ],
  "scoring_rules": "每答出一个采分点得对应分数，同义表达给分，超出参考答案但合理的表述由教师校准"
}
```

校验：

- `points` 非空，每个 `score > 0`。
- `sum(points.score) == max_score`。
- `max_score` 不超过题目分值；没填就用默认值（选择题 3，主观题 6）。
- 分数允许小数，步长 0.5。
- 不合法先带错误信息重试一次；仍不合法，用 `questions.answer_points` 等分兜底，标记 `needs_review=true`。

缓存：按 `question_id` 找 `rubrics.active=true` 的最新版本。没有就生成并写库。教师校准会生成新版本，旧版本置为 `active=false`。一次批改运行可以指定 `requested_rubric_version_id`，用于重批和 A/B 对比。

### 6.5 grade 节点

主观题输入：材料、题干、参考答案、解析、评分细则、学生答案。

输出 JSON：

```json
{
  "score": 4,
  "confidence": 0.82,
  "point_results": [
    {"point_id": "p1", "hit": true, "score": 2, "evidence": "学生写到《独立报》用片面数据", "comment": "意思完整"},
    {"point_id": "p2", "hit": false, "score": 0, "evidence": "", "comment": "没有提到《纽约时报》"},
    {"point_id": "p3", "hit": true, "score": 2, "evidence": "学生提到倾向性标题", "comment": "同义表达"}
  ],
  "rationale": "学生答出两个采分点，第二个缺失，给4分",
  "needs_review": false
}
```

校验：

- 分数等于各采分点得分之和。
- `0 <= score <= max_score`。
- `point_id` 必须来自当前细则，不能新增。
- `confidence` 在 0 到 1 之间。
- `needs_review` 取 LLM 的建议值，服务层再用规则覆盖。

置信度规则：

- 选择题固定 1.0。
- 主观题取 LLM 的 confidence。
- JSON 修复过一次，减 0.1。
- 分数是 0 或满分且答案长度少于 20 字，减 0.15。
- 细则兜底生成，减 0.2。
- 低于阈值 0.75 进入复核。

### 6.6 error_analysis 子图

只在得分小于满分时运行。

输入：题目、材料、细则、学生答案、评分结果。

输出 JSON：

```json
{
  "primary_error_type": "incomplete_summary",
  "secondary_error_types": ["expression"],
  "evidence": "学生只答出第一个采分点，后两个没有展开",
  "suggestion": "练习分点概括，先找全信息再组织语言"
}
```

错因分类固定为以下九类，LLM 只能从中选：

| code | 名称 | 说明 |
|---|---|---|
| `info_filter` | 信息筛选错误 | 找错原文信息或遗漏关键句 |
| `concept_misread` | 概念理解偏差 | 对文中概念理解错误 |
| `incomplete_summary` | 概括不全 | 采分点缺失或概括不完整 |
| `argument_analysis` | 论证分析错误 | 论点、论据、论证方法判断错误 |
| `evidence_misuse` | 论据使用不当 | 引用或分析论据出错 |
| `expression` | 表达不规范 | 语言组织、条理、术语问题 |
| `off_topic` | 答非所问 | 回答与题干要求不符 |
| `blank` | 未作答 | 答案为空 |
| `mcq_wrong` | 选择题误选 | 选择题选了错误选项 |

校验失败时用启发式兜底：答案为空 -> `blank`；选择题 -> `mcq_wrong`；其他 -> `incomplete_summary`。

### 6.7 复核与恢复

复核触发条件，满足任意一条即进入队列：

- `needs_review=true`。
- `confidence < 0.75`。
- 主观题得分是 0 或满分，且答案长度少于 20 字。
- 评分 JSON 经过修复或兜底。
- 教师显式要求复核。

`human_review` 节点调用 `interrupt()`，payload 包含：

```json
{
  "grading_run_id": 12,
  "submission_id": 3,
  "items": [
    {
      "result_id": 45,
      "question_number": 4,
      "question_type": "subjective",
      "student_answer": "...",
      "ai_score": 4,
      "max_score": 6,
      "confidence": 0.68,
      "rationale": "...",
      "point_results": [...]
    }
  ]
}
```

教师通过 `POST /api/grading-runs/{id}/review` 提交：

```json
{
  "corrections": [
    {
      "result_id": 45,
      "final_score": 5,
      "instruction": "学生答出了第三个采分点的同义表达，给分",
      "new_rationale": "补充给分理由"
    }
  ]
}
```

服务层用同一个 `thread_id` 调用 `Command(resume={"corrections": [...]})`。`apply_corrections` 更新 `grading_results.ai_score` 保留原分，写 `final_score`、`reviewed=true`，并把教师原话、旧分、新分、旧理由、新理由写入 `teacher_corrections`。之后图继续到报告节点。

每个节点结束时更新 `grading_runs.heartbeat_at`。如果服务重启，checkpoint 还在，`awaiting_review` 的运行可以继续恢复。启动时把状态为 `running` 且 `heartbeat_at` 超过 30 分钟没有更新的运行标为 `failed`，避免僵尸任务。

### 6.8 自然语言校准

接口：`POST /api/exams/{exam_id}/calibrate`

请求：

```json
{
  "question_id": 12,
  "instruction": "第三点只要提到‘倾向性标题’就给分，不要求写出两家报纸",
  "re_grade_all": true
}
```

流程：

1. 读当前 active 细则。
2. 调 rubric 子图，把教师指令作为额外约束，生成新细则。
3. 校验分数和采分点。
4. 写新 `rubrics` 版本，旧版本 `active=false`，写 `calibration_logs`。
5. `re_grade_all=false` 时只返回新细则；`true` 时查该试卷所有 `completed` 或 `awaiting_review` 的答卷，逐份创建新的 `grading_runs`，使用新细则版本。批量任务用 `asyncio` 信号量限制并发，接口立即返回 run id 列表。

校准不修改原始题库的 `answer_points`，只改 `rubrics`。需要回滚时把旧版本重新置为 active。

## 7. 数据库设计

现有表保持不动：`exams`、`passages`、`questions`、`import_runs`。

新增表写在 `db/schema.sql`，和现有表放在同一个 `public` schema。LangGraph 的 checkpoint 表单独放 `graph` schema，避免和业务表混在一起。

### 7.1 班级与学生

```sql
CREATE TABLE classes (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    grade TEXT,
    subject TEXT NOT NULL DEFAULT '语文',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE students (
    id BIGSERIAL PRIMARY KEY,
    class_id BIGINT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    student_no TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 7.2 答卷与学生答案

```sql
CREATE TABLE submissions (
    id BIGSERIAL PRIMARY KEY,
    exam_id BIGINT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    student_id BIGINT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'api'
        CHECK (source IN ('api', 'synthetic', 'import')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'grading', 'awaiting_review', 'completed', 'failed')),
    total_score NUMERIC(6,1),
    max_score NUMERIC(6,1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    graded_at TIMESTAMPTZ,
    UNIQUE (exam_id, student_id)
);

CREATE TABLE student_answers (
    id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    selected_options JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer_text TEXT,
    synthetic_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (submission_id, question_id)
);
```

一期一个学生一场考试只能有一份答卷。重批不新建 submission，只新建 `grading_runs`。

### 7.3 评分细则

```sql
CREATE TABLE rubrics (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    points JSONB NOT NULL,
    total_score NUMERIC(5,1) NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('ai', 'teacher', 'import')),
    instruction TEXT,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (question_id, version)
);

CREATE UNIQUE INDEX uq_rubrics_active
    ON rubrics (question_id)
    WHERE active;
```

`points` 结构：

```json
[
  {"id": "p1", "description": "...", "score": 2, "evidence_hint": "..."}
]
```

### 7.4 批改运行与结果

```sql
CREATE TABLE grading_runs (
    id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    rubric_version_id BIGINT REFERENCES rubrics(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'awaiting_review', 'completed', 'failed')),
    graph_thread_id TEXT NOT NULL,
    model TEXT,
    prompt_version TEXT,
    error TEXT,
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_grading_runs_submission ON grading_runs (submission_id, created_at DESC);

CREATE TABLE grading_results (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES grading_runs(id) ON DELETE CASCADE,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    rubric_id BIGINT REFERENCES rubrics(id) ON DELETE SET NULL,
    ai_score NUMERIC(5,1) NOT NULL,
    max_score NUMERIC(5,1) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    needs_review BOOLEAN NOT NULL DEFAULT false,
    rationale TEXT,
    point_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    knowledge_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    error_type TEXT,
    error_evidence TEXT,
    final_score NUMERIC(5,1),
    reviewed BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, question_id)
);
```

`ai_score` 保留模型原分，`final_score` 是教师修正后的最终分。没复核时 `final_score = ai_score`。

同一份 submission 可以有多个 grading_runs（重复批改、校准重批）。`submissions.status`、`total_score`、`GET /api/submissions/{id}`、班级/学生报告都以该 submission 最新的 `status='completed'` 的 run 为准；历史 run 保留用于对比和坏案例分析。没有 completed run 时，submission 展示最新 run 的状态，但不计入报告聚合。

`grading_runs.rubric_version_id` 存本次运行请求的细则覆盖版本；普通运行可以为空。每道题实际使用的细则以 `grading_results.rubric_id` 为准。

`grading_results.knowledge_points` 从 structure 子图带下来，报告用它统计薄弱知识点。

### 7.5 教师修正与校准

```sql
CREATE TABLE teacher_corrections (
    id BIGSERIAL PRIMARY KEY,
    result_id BIGINT NOT NULL REFERENCES grading_results(id) ON DELETE CASCADE,
    teacher_name TEXT,
    instruction TEXT,
    old_score NUMERIC(5,1),
    new_score NUMERIC(5,1),
    old_rationale TEXT,
    new_rationale TEXT,
    old_rubric JSONB,
    new_rubric JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE calibration_logs (
    id BIGSERIAL PRIMARY KEY,
    exam_id BIGINT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    instruction TEXT NOT NULL,
    old_rubric_id BIGINT REFERENCES rubrics(id),
    new_rubric_id BIGINT REFERENCES rubrics(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`teacher_corrections` 同时是数据飞轮的原始记录。本期不导出训练集，但字段足够以后生成 JSONL。

### 7.6 错因与报告

```sql
CREATE TABLE error_categories (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE reports (
    id BIGSERIAL PRIMARY KEY,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('class', 'student', 'exam')),
    scope_id BIGINT NOT NULL,
    exam_id BIGINT REFERENCES exams(id) ON DELETE CASCADE,
    content_markdown TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 7.7 默认分值

原始题库没有分值。导入后运行一次 `scripts/init_scores.py`：

- 选择题：`max_score = 3`。
- 主观题：`max_score = 6`。
- 已有非空值不覆盖。

这是本期的假设，不是高考官方分值。教师可以通过校准接口调整。

## 8. API 设计

统一前缀 `/api`，FastAPI 自动生成 `/docs` 和 `/openapi.json`。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/api/exams` | 试卷列表，带题量、已导入年份 |
| GET | `/api/exams/{exam_id}` | 试卷、材料、题目、当前细则 |
| POST | `/api/exams/{exam_id}/submissions` | 创建学生答卷 |
| GET | `/api/submissions/{submission_id}` | 答卷和答案 |
| POST | `/api/submissions/{submission_id}/grade` | 启动批改，返回 `grading_run_id` |
| GET | `/api/grading-runs/{run_id}` | 运行状态、逐题结果、复核队列 |
| POST | `/api/grading-runs/{run_id}/review` | 提交教师修正，恢复图 |
| POST | `/api/exams/{exam_id}/calibrate` | 自然语言校准，可选全量重批 |
| GET | `/api/questions/{question_id}/rubrics` | 细则版本列表 |
| GET | `/api/classes/{class_id}/report?exam_id=` | 班级报告 |
| GET | `/api/students/{student_id}/report?exam_id=` | 学生报告 |
| POST | `/api/dev/seed-synthetic` | 开发环境生成模拟班级和答卷 |
| GET | `/api/dev/bad-cases?exam_id=` | 低置信度、教师改分的坏案例列表 |

请求和响应用 Pydantic 模型，字段名用 snake_case。错误返回 `{"detail": "..."}`，状态码用 400/404/409/500。

批改是后台任务：`POST /grade` 立即返回 `202` 和 `grading_run_id`，客户端轮询 `GET /grading-runs/{id}`。`status=awaiting_review` 时返回复核项；提交复核后再次轮询到 `completed`。

示例流程：

```bash
curl -X POST localhost:8000/api/exams/1/submissions \
  -H "Content-Type: application/json" \
  -d '{"student_id":1,"answers":[{"question_id":1,"selected_options":["C"]},{"question_id":4,"answer_text":"..."}]}'

curl -X POST localhost:8000/api/submissions/1/grade

curl localhost:8000/api/grading-runs/1

curl -X POST localhost:8000/api/grading-runs/1/review \
  -H "Content-Type: application/json" \
  -d '{"corrections":[{"result_id":4,"final_score":5,"instruction":"第三点同义表达给分"}]}'
```

## 9. 服务层与文件结构

```
app/
  __init__.py
  main.py                       FastAPI 入口、lifespan、路由注册
  static/
    index.html                  静态教师复核页，mock 数据
  core/
    config.py                   pydantic-settings，读 .env
    logging.py                  结构化日志
  db/
    session.py                  async engine、session factory
    models.py                   SQLAlchemy 模型
    repositories.py             查询和写入函数
  api/
    deps.py                     依赖注入
    routers/
      exams.py
      submissions.py
      grading.py
      review.py
      calibration.py
      reports.py
      dev.py
  schemas/
    exam.py
    submission.py
    grading.py
    rubric.py
    report.py
  graph/
    state.py                    TypedDict 状态
    builder.py                  顶层图和子图编译
    checkpointer.py             AsyncPostgresSaver 生命周期
    nodes/
      load_submission.py
      prepare_questions.py
      fan_out.py
      collect_results.py
      persist_results.py
      review_gate.py
      human_review.py
      apply_corrections.py
      build_report.py
      persist_report.py
      finalize.py
    subgraphs/
      structure.py
      rubric.py
      error_analysis.py
      grade_question.py
  services/
    grading.py                  启动、恢复、查询运行
    calibration.py              细则校准和全量重批
    reports.py                  班级/学生报告
    synthetic.py                模拟答卷
    llm.py                      DeepSeek 客户端、JSON 校验、重试
    observability.py            Langfuse 包装
  prompts/
    structure.md
    rubric.md
    grade.md
    error_analysis.md
    student_report.md
    class_report.md
    faithfulness.md
scripts/
  init_db.py                    应用数据库 schema
  init_scores.py                填默认分值
  seed_synthetic.py             生成模拟数据
  evaluate.py                   跑黄金样本，输出指标
tests/
  conftest.py
  unit/
  integration/
  fixtures/
    golden_grading.jsonl
```

静态页已经有一个可用版本，位于 `app/static/index.html`。可以直接打开，也可以用 `.venv\Scripts\python.exe -m http.server 8000 --directory app\static` 起一个本地服务。页面里的 `examData` 是 mock 数据，接口接上后改成 `fetch /api/...`。已经在 Edge 里按 1440px、390px 和深色模式检查过，复核提交后的状态、待复核数和队列更新也测过。

根目录 `config.py` 保留兼容入口，内容改为从 `app.core.config` 导出 `settings`。

## 10. LLM 客户端

统一接口：

```python
class LLMClient:
    async def complete_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        temperature: float = 0.1,
        max_retries: int = 1,
        trace_name: str,
    ) -> BaseModel: ...
```

实现：

- 用 `openai.AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)`。
- 请求带 `response_format={"type": "json_object"}`。
- 从返回内容中提取 JSON，先直接 `json.loads`，失败再尝试去掉 markdown 代码块和截取第一个 `{` 到最后一个 `}`。
- 用 Pydantic 校验；失败把错误信息拼进修复 prompt，重试一次。
- 仍失败抛 `LLMOutputError`，调用方决定兜底或标记复核。
- 超时 60 秒，网络错误重试 2 次，指数退避。
- 并发用 `asyncio.Semaphore(settings.grading_concurrency)`，默认 5。
- 所有调用经过 `langfuse.openai.AsyncOpenAI` 包装；如果本地 Langfuse 版本不兼容，退回 `@observe` 手动包 span，功能不受影响。
- 成本控制：同一题同一细则版本的评分结果不缓存，因为学生答案不同；细则按 `question_id + 版本` 缓存，避免重复生成。

## 11. Langfuse 与评测

### 11.1 Trace 结构

每次批改一条 trace：

- trace 名称：`grade_submission`
- metadata：`exam_id`、`submission_id`、`grading_run_id`、`model`、`prompt_version`
- span：`load_submission`、`structure`、`rubric`、`grade`、`error_analysis`、`report`
- generation：每次 DeepSeek 调用，带 prompt、completion、token、耗时
- score：
  - `json_valid`：0 或 1
  - `task_completed`：0 或 1
  - `needs_review`：0 或 1
  - `teacher_override`：复核后 0 或 1
  - `score_mae`：评测脚本写入
  - `faithfulness`：评测脚本写入

### 11.2 评测脚本

`scripts/evaluate.py`：

1. 读 `tests/fixtures/golden_grading.jsonl`，每行包含题目、学生答案、期望分数、期望错因。
2. 用当前活跃细则逐条评分，不写业务库，只写 Langfuse trace。
3. 输出：
   - JSON 有效率
   - 分数平均绝对误差（MAE）
   - 完全一致率
   - 错因分类准确率
   - 平均置信度
   - 需复核比例
4. 可选 `--faithfulness`：抽 10 条评分理由，用 LLM judge 判断理由是否有原文或参考答案依据，写 `faithfulness` score。

黄金样本先手工标 20 条，覆盖 3 道选择题、4 道主观题、满分/部分/错答/空白。样本不放进生产库。

### 11.3 坏案例分析

`GET /api/dev/bad-cases?exam_id=` 返回：

- `confidence < 0.6` 的结果。
- 教师改分幅度大于等于 2 分的结果。
- JSON 修复或兜底的结果。
- 同一题同一错因重复出现的学生答案。

每条带题目、学生答案、AI 分数、最终分数、理由、错因。用于人工复盘和后续 prompt 迭代。

## 12. 模拟数据

`scripts/seed_synthetic.py` 参数：

- `--exam-code`，默认 `2023-新高考Ⅰ卷`
- `--students`，默认 10
- `--class-name`，默认 `高二(1)班`
- `--seed`，默认 42

生成规则：

- 一个班、10 个学生，姓名用 `模拟学生01` 到 `模拟学生10`。
- 每个学生一份答卷，覆盖该试卷所有题。
- 选择题：70% 选正确答案，30% 随机选一个错误选项。
- 主观题四种画像：
  - `full`：直接用参考答案。
  - `partial`：取参考答案前一半采分点。
  - `wrong`：写一句与题干相关但答非所问的话。
  - `blank`：空字符串。
- 分配比例：full 30%、partial 40%、wrong 20%、blank 10%。
- 每题单独抽样，用固定随机种子保证可重复。
- `student_answers.synthetic_profile` 记录画像，方便评测分组。

默认不调用 LLM，保证脚本可重复、无 API 成本。需要更真实的改写样本时，后续加 `--paraphrase` 开关。

## 13. 测试策略

- 单元测试：
  - JSON 提取和修复。
  - 评分细则分数校验。
  - 置信度规则和复核触发。
  - 错因兜底。
  - 报告聚合计算。
- 集成测试：
  - 测试库 `Mark_test`，fixture 执行 `db/schema.sql`。
  - 用 `FakeLLMClient` 返回固定 JSON，跑完整顶层图和 API。
  - 覆盖：选择题满分/错选、主观题满分/部分/空白、复核恢复、校准新版本、报告。
- 真实 API 测试：
  - 标 `@pytest.mark.live`，只跑 1 道主观题，需要 `.env` 里的 key。
  - 默认不跑，避免费用和波动。
- 端到端验收：
  - 启动服务，生成模拟数据，批改一份答卷，查看 Langfuse trace，完成一次复核，生成班级报告。
- 代码质量：
  - `ruff check .`
  - `ruff format --check .`
  - `pytest -q`

测试库创建命令：

```bash
docker exec agent_test-postgres createdb -U postgres Mark_test
```

测试结束后不删库，fixture 每次清表。

## 14. 依赖与运行

`pyproject.toml` 增加依赖：

- `fastapi`
- `uvicorn[standard]`
- `sqlalchemy[asyncio]`
- `psycopg[binary,pool]`
- `pydantic`
- `pydantic-settings`
- `langgraph`
- `langgraph-checkpoint-postgres`
- `langchain-core`
- `openai`
- `langfuse`
- `pytest`
- `pytest-asyncio`
- `httpx`
- `ruff`

版本在实现时按 PyPI 当前稳定版锁定，写进 `pyproject.toml`。

首次安装：

```powershell
.venv\Scripts\python.exe -m ensurepip --upgrade
.venv\Scripts\python.exe -m pip install -e .
```

运行：

```powershell
.venv\Scripts\python.exe scripts\run_server.py --port 8000
```

初始化：

```powershell
.venv\Scripts\python.exe scripts\init_db.py
.venv\Scripts\python.exe scripts\init_scores.py
.venv\Scripts\python.exe scripts\seed_synthetic.py --exam-code 2023-新高考Ⅰ卷
```

## 15. 配置项

`app/core/config.py` 的 `Settings`：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MARK_DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/Mark` | 业务库 |
| `API_KEY` | 无 | DeepSeek key，从 `.env` 读 |
| `BASE_URL` | `https://api.deepseek.com` | DeepSeek 兼容地址 |
| `MODEL_NAME` | `deepseek-chat` | 模型 |
| `LLM_TIMEOUT_SECONDS` | 60 | 单次调用超时 |
| `LLM_MAX_RETRIES` | 2 | 网络重试 |
| `GRADING_CONCURRENCY` | 5 | 逐题并行上限 |
| `REVIEW_CONFIDENCE_THRESHOLD` | 0.75 | 复核阈值 |
| `DEFAULT_MCQ_SCORE` | 3 | 选择题默认分 |
| `DEFAULT_SUBJECTIVE_SCORE` | 6 | 主观题默认分 |
| `LANGFUSE_ENABLED` | true | 是否上报 |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse 地址 |
| `LANGFUSE_PUBLIC_KEY` | 无 | Langfuse key |
| `LANGFUSE_SECRET_KEY` | 无 | Langfuse key |
| `PROMPT_VERSION` | `v1` | 写进 trace 和运行记录 |

`.env` 继续忽略。另建 `.env.example`，只放变量名和占位值。

## 16. 里程碑

### M1：基础层

- 装依赖，建 `app/` 包，`Settings` 读 `.env`。
- 扩展 `db/schema.sql`，写 SQLAlchemy 模型和 session。
- FastAPI 启动，`/health`、`/api/exams`、`/api/exams/{id}` 可用。
- `scripts/init_db.py`、`scripts/init_scores.py`。
- 单元测试和健康检查通过。

验收：服务能起，能查到 21 个试卷组和 69 道题，测试通过。

### M2：批改链路

- `LLMClient`、JSON 校验和重试。
- 三个子图和顶层图，选择题走短路，主观题走完整流程。
- `grading_runs`、`grading_results` 落库。
- 模拟数据脚本。
- 集成测试用 `FakeLLMClient` 跑通一份答卷。

验收：生成 10 份模拟答卷，批改 2023 新高考Ⅰ卷的一份，该卷 5 道题都有结果，选择题分数正确，主观题有采分点和理由。题库全量 69 道题的结构和主观题细则生成另做一次冒烟测试，不作为 M2 的门槛。

### M3：人机协同与报告

- 复核触发、`interrupt`、`Command(resume=...)`。
- 教师修正落库。
- 自然语言校准、新细则版本、全量重批。
- 学生报告和班级报告接口。
- 坏案例接口。

验收：低置信度题进入复核，提交修正后运行完成；校准生成新细则并能重批；报告包含分数、薄弱点和建议。

### M4：观测与评测

- Langfuse trace、span、score 接入。
- `scripts/evaluate.py` 和黄金样本。
- 文档和运行说明。

验收：Langfuse 能看到一次完整批改的 trace；评测脚本输出 MAE、JSON 有效率、错因准确率；README 能按步骤复现。

## 17. 风险与假设

- **没有逐题分值。** 用 3/6 默认分，教师可改。报告和指标都基于这个假设，不能当作官方分数。
- **没有真实学生答卷。** 模拟答卷只能验证链路和格式，不能证明真实场景的评分准确率。
- **DeepSeek JSON 不稳定。** 靠 JSON 模式、Pydantic 校验、一次修复和兜底控制；失败项标记复核，不静默丢数据。
- **主观题评分标准有争议。** 采分点来自参考答案和解析，可能不是官方细则。教师校准是必要环节。
- **后台任务不持久。** FastAPI 后台任务在进程重启时会丢；`awaiting_review` 的 checkpoint 能恢复，`running` 的会被标记失败。生产环境需要换持久队列。
- **Langfuse SDK 兼容性。** 本地服务是 Langfuse 4；Python SDK 版本需要实测，不兼容就退回 REST 或 `@observe`。
- **成本。** 一道主观题至少一次细则生成（首次）加一次评分，错题再加一次错因。10 份答卷的演示成本可控；全量重批要加并发上限和 dry-run。
- **没有鉴权。** 只能本地运行，不暴露公网。
- **长文本。** 单篇材料 1-3 千字，DeepSeek 上下文足够；如果以后加整卷多材料，需要改成分段或摘要。

## 18. 验收标准

- `scripts/run_server.py` 启动，`/health` 返回 200，`/docs` 可打开。
- 打开 `app/static/index.html` 能看到 2023 新高考Ⅰ卷的阅读材料、5 道题、班级 KPI、复核队列和错因分布；提交复核后，题目状态、待复核数和复核队列同步更新。
- `GET /api/exams` 返回 21 个试卷组；`GET /api/exams/1` 返回 5 道题和阅读材料。
- 生成 10 个学生和答卷，批改一份，`grading_results` 有该卷 5 条结果，选择题分数与答案一致。
- 主观题结果包含 `point_results`、`rationale`、`confidence`；低置信度进入复核。
- 提交复核后 `grading_runs.status=completed`，`teacher_corrections` 有记录，`final_score` 更新。
- 自然语言校准生成新的 `rubrics` 版本，旧版本 `active=false`；`re_grade_all=true` 时创建新的 `grading_runs`。
- 班级报告包含平均分、分数段、错因分布、薄弱点、建议；学生报告包含逐题结果和建议。
- Langfuse 能看到 `grade_submission` trace，包含子图 span 和 DeepSeek generation。
- `pytest -q` 通过；`ruff check .` 通过。
- `.env` 没有被 git 跟踪；代码里没有明文 key。
