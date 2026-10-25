"""Attempted-question audit only; never used to rank or select questions."""
from threading import Lock


class QuestionHistory:
    def __init__(self):
        self._lock = Lock()
        self._questions = {}

    def add(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("history entries must be nonempty question strings")
        key = " ".join(question.split()).casefold()
        with self._lock:
            self._questions[key] = min(question.strip(), self._questions.get(key, question.strip()))

    def snapshot(self):
        with self._lock:
            return [self._questions[key] for key in sorted(self._questions)]

    def clear(self):
        with self._lock:
            self._questions.clear()


class RecordingAnalysis:
    def __init__(self, engine, history):
        self.engine, self.history = engine, history

    def analyze_initial(self, **kwargs):
        self.history.add(kwargs["question"].question)
        return self.engine.analyze_initial(**kwargs)

    def analyze_follow_up(self, **kwargs):
        self.history.add(kwargs["follow_up"].question)
        return self.engine.analyze_follow_up(**kwargs)
