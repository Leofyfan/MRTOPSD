from verl.utils.reward_score.math_dapo import compute_score


def test_compute_score_accepts_last_boxed_answer():
    result = compute_score(
        solution_str=(
            "We solve the problem step by step.\n"
            "Some intermediate reasoning.\n\n"
            "### Final Answer\n\n"
            "$$\n\\boxed{\\sqrt{41}}\n$$"
        ),
        ground_truth="\\sqrt{41}",
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "\\sqrt{41}"


def test_compute_score_falls_back_to_answer_prefix_when_no_boxed_answer():
    result = compute_score(
        solution_str="Reasoning omitted.\nAnswer: -i",
        ground_truth="-i",
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "-i"
