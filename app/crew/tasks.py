from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkItem:
    task_id: str
    task_text: str
    # If set, this task depends on the deliverable produced by these earlier
    # task_ids. Used to score cross-task consistency (memory vs cold).
    depends_on: list[str] = field(default_factory=list)


WORK_ITEMS: list[WorkItem] = [
    WorkItem(
        "syllabus",
        "Define the 6-week syllabus for a course on Transformer attention: name each "
        "week's theme and the single core skill it teaches. Pick concrete week themes.",
    ),
    WorkItem(
        "module_outline",
        "Design a detailed module outline for Week 1 (the first week of the syllabus), "
        "consistent with the syllabus you defined.",
        depends_on=["syllabus"],
    ),
    WorkItem(
        "quiz_bank",
        "Write 10 assessment questions with an answer key for Week 1, matching the Week 1 "
        "module outline.",
        depends_on=["module_outline"],
    ),
    WorkItem(
        "lab_guide",
        "Create lab instructions for implementing a toy attention block, aligned with the "
        "Week 1 skills.",
        depends_on=["module_outline"],
    ),
    WorkItem(
        "grading_rubric",
        "Draft a grading rubric with 4 proficiency levels for the attention lab. It MUST "
        "grade exactly the deliverables the lab guide asks students to produce.",
        depends_on=["lab_guide"],
    ),
    WorkItem(
        "misconceptions",
        "List the top student misconceptions about attention and how to correct each, "
        "tied to the Week 1 themes.",
        depends_on=["module_outline"],
    ),
    WorkItem(
        "faq",
        "Write FAQ answers for common student support tickets about the course, consistent "
        "with the syllabus and lab.",
        depends_on=["syllabus", "lab_guide"],
    ),
    WorkItem(
        "capstone",
        "Define a capstone project brief. It MUST build on the exact week themes from the "
        "syllabus and reuse the rubric's proficiency levels.",
        depends_on=["syllabus", "grading_rubric"],
    ),
]


# The single most cinematic "it remembered" probe: it explicitly references a
# decision (the Week 3 theme) made many tasks earlier.
RECALL_PROBE = WorkItem(
    "recall_probe",
    "Without re-deriving anything, state the Week 3 theme and core skill from the syllabus "
    "you defined earlier, then design one in-class exercise for exactly that theme.",
    depends_on=["syllabus"],
)
