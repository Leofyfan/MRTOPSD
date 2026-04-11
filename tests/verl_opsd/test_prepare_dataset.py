from verl_opsd.prepare_dataset import build_student_messages, convert_split


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        if tokenize:
            return list(range(len(messages[0]["content"].split())))
        return messages[0]["content"]


def test_convert_split_adds_problem_id_and_preserves_problem_fields():
    dataset = [
        {"global_row_idx": 10, "problem": "2+2?", "solution": "4", "answer": "4", "source": "toy"},
        {"global_row_idx": 11, "problem": "3+3?", "solution": "6", "answer": "6", "source": "toy"},
    ]
    records = convert_split(dataset, DummyTokenizer(), max_student_prompt_length=999, max_teacher_prompt_length=999)

    assert records[0]["problem_id"] == "opsd-problem-10"
    assert records[0]["problem"] == "2+2?"
    assert records[0]["answer"] == "4"
    assert records[0]["extra_info"]["problem_id"] == "opsd-problem-10"

    assert records[1]["problem_id"] == "opsd-problem-11"
    assert records[1]["problem"] == "3+3?"
    assert records[1]["answer"] == "6"
    assert records[1]["extra_info"]["problem_id"] == "opsd-problem-11"


def test_student_prompt_requires_boxed_answer_on_last_line():
    prompt = build_student_messages("2+2?")[0]["content"]

    assert "boxed" in prompt
    assert "last line" in prompt
    assert "Do not write anything after the boxed answer" in prompt
