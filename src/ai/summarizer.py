"""LLM-based course lecture summarization via ModelScope API."""

import time

from openai import APITimeoutError, OpenAI

from src.runtime import config
from src.ai.study_notes import STUDY_PROMPT, format_notes

SYSTEM_PROMPT= r"""你是一个专业的课程助教。你的任务是根据用户提供的课程录音文本和ppt文字ocr部分，生成用于学生自学和期末复习的详细笔记。
1. **直接输出**：不要包含任何"好的"、"没问题"、"以下是总结"等客套话，不要输出全局课程名称大标题（由系统自动生成），直接开始总结即可。
2. **文本清洗**：语言必须通顺、逻辑清晰，严格去除口语化表达、重复句和无意义的录音识别错误等。内容可能被识别成同音字，需要通过学术语境修复。
3. **格式严格**：
   - 必须使用 Markdown 格式排版。
   - 标题结构清晰与级别限制：只允许使用三级及后续级别的标题（即只能使用`###`、`####`或`#####`），禁止使用 `#` 和 `##`。用清晰的标题组织结构。
   - 不得使用超过两级的缩进和超过两级嵌套的bullet point。
   - 尽可能用完整的段落来组织老师的讲解，适量使用bullet point，不得过度使用bullet point
   - 合理使用加粗、列表、表格、分级标题、段首小标题等形式来组织信息，确保结构清晰。
4. **公式规范**：所有数学公式或科学变量必须使用规范的 LaTeX 语法（行内公式用 $...$，行间公式用 $$...$$）。由于图床限制，latex公式中不要出现中文。
5. **忠于原文与详略得当**：总结必须有详有略，总体以详细风格为主。笔记正文篇幅参考用户提供的目标范围，该范围仅按音频转录长度计算，不包含证据表、课首主线和板块过渡句；不为凑字数重复或扩写。PPT 用于补充公式、定义和案例，其文字数量不增加目标篇幅。
包含具体的推导细节、案例、文献或者核心概念，不要过度概括，也不要仅仅原文复述。禁止捏造录音中未提及的内容。
6. 你需要格外注意课程中是否提及了作业、考试、签到、组队等关键事项，如果有的话，用三级标题【课程事项提醒】标注在开头。
7. **文风示例**：以下是一个关于"梯度下降"的片段，展示了笔记总结过程中【错误的】和【正确的】的两种总结风格，请严格模仿后者。

【❌ 错误的风格】
## 梯度下降

**定义：**
- 梯度下降是一种优化算法
- 用于最小化损失函数
- 广泛应用于机器学习

**核心步骤：**
- 计算梯度
- 更新参数
- 重复迭代

**学习率：**
- 学习率决定步长
- 太大会发散
- 太小会收敛慢
- 需要调参

**类型：**
- 批量梯度下降（BGD）
- 随机梯度下降（SGD）
- 小批量梯度下降（Mini-batch GD）

---

【✅ 正确的风格】

### 梯度下降

梯度下降是最小化损失函数 $L(\theta)$ 的核心优化算法。其基本思想是沿着损失函数对参数 $\theta$ 的梯度的反方向迭代更新，每一步的更新公式为 $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$，其中 $\eta$ 称为学习率，控制每次更新的步长大小。

**学习率的选取至关重要**：若 $\eta$ 过大，参数更新幅度过猛，损失函数可能在最优点附近震荡甚至发散；若 $\eta$ 过小，收敛速度极慢，训练成本大幅上升。实践中通常通过**学习率调度（learning rate schedule）**或**自适应方法（如 Adam）**来缓解这一问题。

根据每次更新时使用的样本量，梯度下降可分为三类：**批量梯度下降（BGD）** 每次使用全部训练数据，梯度估计准确但计算开销大；**随机梯度下降（SGD）** 每次仅用单个样本，更新频繁但噪声大；**小批量梯度下降（Mini-batch GD）** 则折中两者，是深度学习中最常用的形式。
---

8. **输入材料格式**：
   - 你收到的输入可能是两种格式之一：
     A) **带时间轴的分段格式**：用 `=== 时间段 mm:ss – mm:ss ===` 分隔每个 10 分钟段；段内有【音频转录】和【PPT 文字识别】两类内容。
     B) **平铺格式**：先一段【音频转录（无时间轴）】，后一段【PPT 文字识别（按出现顺序）】，每张 PPT 带 `[页 N @ mm:ss]` 标签。
   - 无论哪种格式，**PPT 文字是真实的板书 / 课件内容**，比音频识别噪声更可靠；遇到音频识别错的术语（同音字、专有名词），优先以 PPT 文字为准修正，但PPT的文字也可能有ocr的错误。
   - 但是，你的总结的组织的主线仍然应该以讲师的讲解为逻辑组织，同时把PPT的信息补充进去。
   - 同时，需要注意的是，由于该平台以屏幕截图的方式记录ppt，因此，尽管我们提供的ocr版本已经尽最大努力进行了清洗，你收到的ppt仍然可能有无关网页、导航栏、桌面或系统页面等无关噪音。这些噪音不应被用于课程内容的总结。
   - 输出仍按之前的格式要求，不要保留时间戳标签，只把这些信息当作上下文辅助理解，不用说哪些来自转写哪些来自ppt，自然地合并录音转写和ppt中的知识，生成高质量笔记。"""

def target_note_chars(transcript_chars: int) -> int:
    """Round half up; OCR volume must never affect the writing budget."""
    if transcript_chars < 0:
        raise ValueError("transcript_chars must be non-negative")
    return min(8000, max(500, (transcript_chars + 2) // 4))


class IncompleteResponse(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Incomplete study notes response: {reason}")


class Summarizer:
    """Course lecture summarizer with multi-provider fallback.

    Iterates config.MODEL_PROVIDERS in declared order. Within each provider,
    tries each model in declared order. Returns the first successful result.
    Setting only DASHSCOPE_API_KEY still works because the default
    MODEL_PROVIDERS list ships a modelscope entry that reads it.
    """

    def __init__(self):
        self.providers = config.resolve_model_providers()
        if not self.providers:
            raise ValueError(
                "No model provider available. "
                "Set at least one provider's API key (e.g. DASHSCOPE_API_KEY)."
            )
        self._clients = {
            p["name"]: OpenAI(api_key=p["api_key"], base_url=p["base_url"])
            for p in self.providers
        }

    @staticmethod
    def _diagnose(model_id, attempt, target, content, t0, response=None,
                  error_type=None):
        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        result = getattr(getattr(choice, "message", None), "content", None)
        reason = getattr(choice, "finish_reason", None)
        # Only log known enum values; arbitrary provider text may contain data.
        known_reasons = {"stop", "length", "content_filter", "tool_calls", "function_call"}
        reason = reason if reason in known_reasons else "unknown"
        usage = getattr(response, "usage", None)
        def token_value(name):
            value = getattr(usage, name, None)
            return value if isinstance(value, int) else "unknown"
        print(
            f"[Summarizer] model={model_id} attempt={attempt}/2 "
            f"target_chars={target} input_chars={len(content)} "
            f"elapsed={time.monotonic() - t0:.1f}s finish_reason={reason} "
            f"body_chars={len(result) if isinstance(result, str) else 'unknown'} "
            f"prompt_tokens={token_value('prompt_tokens')} "
            f"completion_tokens={token_value('completion_tokens')} "
            f"total_tokens={token_value('total_tokens')} "
            f"error_type={error_type or 'none'}", flush=True,
        )

    def _call_llm(self, client: OpenAI, model: str, title: str, content: str,
                  *, target_chars: int, attempt: int, model_id: str) -> str:
        t0 = time.monotonic()
        lower = (target_chars * 85 + 50) // 100
        upper = min(8000, (target_chars * 115 + 50) // 100)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + STUDY_PROMPT},
                    {"role": "user", "content": (
                        f"以下是课程《{title}》的录音转录与 PPT 材料。"
                        f"笔记正文目标约 {target_chars} 字，参考范围 {lower}～{upper} 字。"
                        "此范围不包含证据表、课首主线和板块过渡句；保留主要推导、案例与公式，"
                        "不必凑字数。PPT 仅补充内容，不增加篇幅要求。"
                        f"请输出完整 JSON 对象：\n\n{content}"
                    )},
                ],
                response_format={"type": "json_object"},
                max_tokens=24000,
                timeout=300,
            )
        except Exception as exc:
            self._diagnose(model_id, attempt, target_chars, content, t0,
                           error_type=type(exc).__name__)
            raise
        self._diagnose(model_id, attempt, target_chars, content, t0, response)
        if not response.choices:
            raise IncompleteResponse("empty_choices")
        choice = response.choices[0]
        # Check finish reason first: filtering must never trigger an empty-body retry.
        if choice.finish_reason not in ("stop", None):
            reason = choice.finish_reason
            if reason not in {"length", "content_filter", "tool_calls", "function_call"}:
                reason = "unknown_finish_reason"
            raise IncompleteResponse(reason)
        result = choice.message.content
        if not result or not result.strip():
            raise IncompleteResponse("empty_body")
        if choice.finish_reason != "stop":
            raise IncompleteResponse("unknown_finish_reason")
        try:
            return format_notes(result, content)
        except Exception as exc:
            # Parser exceptions may include snippets of model output. Keep them private.
            raise ValueError(f"Invalid study notes format ({type(exc).__name__})") from None

    def summarize(self, title: str, content: str, *,
                  transcript_chars: int) -> tuple[str, str]:
        """Generate notes with a transcript-only budget and bounded model retries.

        SDK network retries retain their defaults. Each model receives at most
        two application-level attempts; other providers remain fallbacks.
        """
        target = target_note_chars(transcript_chars)
        if not content or not content.strip():
            return ("（内容为空）", "")
        errors = []
        for provider in self.providers:
            client = self._clients[provider["name"]]
            for model in provider["models"]:
                model_id = f"{provider['name']}/{model}"
                model_target = target
                for attempt in (1, 2):
                    try:
                        result = self._call_llm(
                            client, model, title, content, target_chars=model_target,
                            attempt=attempt, model_id=model_id,
                        )
                        return (result, model_id)
                    except Exception as exc:
                        retry = False
                        # Never log arbitrary API exception messages (may echo input).
                        if isinstance(exc, IncompleteResponse):
                            detail = str(exc)
                            retry = exc.reason in {"length", "empty_choices", "empty_body"}
                            if exc.reason == "length":
                                model_target = (model_target * 75 + 50) // 100
                        else:
                            detail = type(exc).__name__
                            retry = isinstance(exc, (APITimeoutError, TimeoutError))
                        print(f"[Summarizer] {model_id} attempt={attempt}/2 failed: "
                              f"{detail}", flush=True)
                        errors.append(f"{model_id} attempt={attempt}: {detail}")
                        if attempt == 2 or not retry:
                            break
        raise RuntimeError("All LLM models failed:\n" + "\n".join(errors))
