from __future__ import annotations

import dspy


class GapPlanner(dspy.Signature):
    """Identify unknown angles from prior memory."""

    topic = dspy.InputField()
    known_facts = dspy.InputField(desc="Bullet list of retrieved memories.")
    open_questions = dspy.OutputField(desc="Five concrete questions not already known.")


class Researcher(dspy.Signature):
    """Generate concise, source-aware findings."""

    topic = dspy.InputField()
    open_questions = dspy.InputField()
    findings = dspy.OutputField(desc="5-8 non-overlapping findings.")


class Synthesizer(dspy.Signature):
    """Combine prior memory and new findings into one deduplicated report."""

    topic = dspy.InputField()
    prior_memory = dspy.InputField()
    new_findings = dspy.InputField()
    report = dspy.OutputField(desc="Structured report with no repeated facts.")
