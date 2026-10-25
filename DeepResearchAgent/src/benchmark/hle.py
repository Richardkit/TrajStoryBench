import re
from typing import Optional, Any, List, Dict
from pydantic import Field, ConfigDict, PrivateAttr

from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK
from src.utils import dedent

SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that solves challenging academic questions across many subjects. Please think step by step, deliver both the reasoning and the result, and strictly follow the provided output format.
    
    Your response should be in the following format:
    The output should be a JSON object with the following fields, DO NOT add any other text like "```json" or "```" or anything else:
    {
        "reasoning": "Your step-by-step explanation",
        "result": "Your final answer"
    }

    Notes:
    - For multiple-choice questions, provide only the letter of the correct option (e.g. "A", "B", "C", "D").
    - For short-answer / exact-match questions, provide a concise answer string.
    - If an image is provided, use it carefully as part of your reasoning.
    
    Please solve the following problem:
""")


def _extract_image_media_type(data_uri: str) -> str:
    """Extract media type from a data URI like 'data:image/jpeg;base64,...'."""
    match = re.match(r"data:(image/\w+);", data_uri)
    if match:
        return match.group(1)
    return "image/png"


@BENCHMARK.register_module(force=True)
class HLEBenchmark(Benchmark):
    """
    Humanity's Last Exam (HLE) Benchmark – a multi-modal benchmark of 2500
    expert-level questions across dozens of academic subjects.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="hle", description="The name of the benchmark")
    path: str = Field(default="datasets/hle", description="The path to the benchmark dataset")

    _data_records: List[Dict] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tasks: List[Task] = PrivateAttr(default_factory=list)

    system_prompt: Optional[str] = Field(default=SYSTEM_PROMPT, description="The system prompt for the benchmark")

    async def initialize(self):
        from src.data.hle import HLEDataset
        dataset = HLEDataset(
            path=self.path,
            name="default",
            split="test"
        )
        if hasattr(dataset, 'data'):
            self._data_records = dataset.data.to_dict(orient="records")
        await self.reset()

    async def reset(self) -> Optional[Task]:
        self._index = 0
        self._tasks = []
        return await self.step()

    async def step(self) -> Optional[Task]:
        if self._index >= len(self._data_records):
            return None

        record = self._data_records[self._index]
        self._index += 1

        question_text = record.get("question", "")
        image_data = record.get("image")
        answer_type = record.get("answer_type", "exactMatch")

        if answer_type == "multipleChoice":
            question_text += "\n\nThis is a multiple-choice question. Provide only the letter of the correct option."

        extra = {
            k: v for k, v in record.items()
            if k not in ["true_answer", "answer", "task_id", "id", "question"]
        }

        if image_data:
            extra["image"] = image_data
            extra["image_media_type"] = _extract_image_media_type(image_data)

        return Task(
            task_id=record.get("task_id", f"{self._index:04d}"),
            input=question_text,
            system_prompt=self.system_prompt,
            ground_truth=record.get("true_answer") or record.get("answer"),
            extra=extra,
        )

    async def eval(self, task: Task) -> Optional[Task]:
        result = str(task.result).strip() if task.result is not None else ""
        ground_truth = str(task.ground_truth).strip() if task.ground_truth is not None else ""

        answer_type = task.extra.get("answer_type", "exactMatch") if task.extra else "exactMatch"

        if answer_type == "multipleChoice":
            result_letter = self._extract_choice(result)
            gt_letter = self._extract_choice(ground_truth)
            task.result = result_letter
            task.ground_truth = gt_letter
            task.score = 1.0 if result_letter and result_letter == gt_letter else 0.0
        else:
            norm_result = self._normalize(result)
            norm_gt = self._normalize(ground_truth)
            task.result = norm_result
            task.ground_truth = norm_gt
            task.score = 1.0 if norm_result and norm_result == norm_gt else 0.0

        self._tasks.append(task)
        return task

    async def stats(self) -> Optional[Stats]:
        total = len(self._data_records)
        attempted = len(self._tasks)
        correct = sum(1 for r in self._tasks if r.score and r.score >= 1.0)

        task_times = {r.task_id: r.time for r in self._tasks if r.time is not None}
        avg_time = sum(task_times.values()) / len(task_times) if task_times else 0.0

        mc_tasks = [t for t in self._tasks if t.extra and t.extra.get("answer_type") == "multipleChoice"]
        em_tasks = [t for t in self._tasks if t.extra and t.extra.get("answer_type") == "exactMatch"]

        mc_correct = sum(1 for t in mc_tasks if t.score and t.score >= 1.0)
        em_correct = sum(1 for t in em_tasks if t.score and t.score >= 1.0)

        return Stats(
            accuracy=correct / attempted if attempted > 0 else 0.0,
            total=total,
            correct=correct,
            wrong=attempted - correct,
            times=task_times,
            average_time=avg_time,
            extra={
                "multiple_choice_accuracy": mc_correct / len(mc_tasks) if mc_tasks else 0.0,
                "exact_match_accuracy": em_correct / len(em_tasks) if em_tasks else 0.0,
                "multiple_choice_total": len(mc_tasks),
                "exact_match_total": len(em_tasks),
            },
        )

    @staticmethod
    def _extract_choice(text: str) -> str:
        """Extract a single letter choice (A-Z) from text."""
        text = text.strip()
        if len(text) == 1 and text.isalpha():
            return text.upper()
        match = re.search(r'\b([A-Z])\b', text.upper())
        if match:
            return match.group(1)
        return text.strip().upper()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize answer text for exact-match comparison."""
        if not text:
            return ""
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip(' .,;:!?')
        return text
