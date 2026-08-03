from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.funaudiochat import funaudiochat2code2wav


def _stage_list(audio_token_ids=None, speech_ids=None):
    output = SimpleNamespace(multimodal_output={})
    if audio_token_ids is not None:
        output.multimodal_output["audio_token_ids"] = audio_token_ids
    if speech_ids is not None:
        output.multimodal_output["speech_ids"] = speech_ids
    stage_output = SimpleNamespace(outputs=[output])
    stage = SimpleNamespace(engine_outputs=[stage_output])
    return [stage]


def test_filters_invalid_audio_tokens_for_code2wav():
    stage_inputs = _stage_list(torch.tensor([-1, 0, 17, 6560, 6561, 7000], dtype=torch.long))

    prompts = funaudiochat2code2wav(stage_inputs, engine_input_source=[0])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0, 17, 6560]
    assert prompts[0]["multi_modal_data"] is None
    assert prompts[0]["mm_processor_kwargs"] is None


def test_accepts_list_audio_tokens():
    stage_inputs = _stage_list([5, 12, 6559, 6561])

    prompts = funaudiochat2code2wav(stage_inputs, engine_input_source=[0])

    assert prompts[0]["prompt_token_ids"] == [5, 12, 6559]


def test_accepts_step_aligned_audio_token_rows():
    stage_inputs = _stage_list(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            dtype=torch.long,
        )
    )

    prompts = funaudiochat2code2wav(stage_inputs, engine_input_source=[0])

    assert prompts[0]["prompt_token_ids"] == [0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_drops_fully_negative_step_aligned_rows():
    stage_inputs = _stage_list(
        torch.tensor(
            [
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
                [21, 22, 23, 24, 25],
            ],
            dtype=torch.long,
        )
    )

    prompts = funaudiochat2code2wav(stage_inputs, engine_input_source=[0])

    assert prompts[0]["prompt_token_ids"] == [0, 0, 0, 0, 0, 21, 22, 23, 24, 25]


def test_prefers_incremental_audio_token_ids_over_cumulative_speech_ids():
    stage_inputs = _stage_list(
        audio_token_ids=torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]], dtype=torch.long),
        speech_ids=torch.tensor([101, 102, 103], dtype=torch.long),
    )

    prompts = funaudiochat2code2wav(stage_inputs, engine_input_source=[0])

    assert prompts[0]["prompt_token_ids"] == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_raises_when_audio_token_ids_are_missing():
    stage_inputs = _stage_list(None)

    with pytest.raises(ValueError, match="speech_ids|audio_token_ids"):
        funaudiochat2code2wav(stage_inputs, engine_input_source=[0])
