"""SciCode-specific chat inferencer."""

from typing import List

from opencompass.datasets.scicode import (extract_candidate_code,
                                          trim_to_parsable_suffix_safe)
from opencompass.registry import ICL_INFERENCERS

from .icl_chat_inferencer import ChatInferencer


@ICL_INFERENCERS.register_module()
class SciCodeChatInferencer(ChatInferencer):
    """Keep SciCode history aligned with the official step-by-step flow."""

    def infer_every(self, chat: List[dict], index: int, output_handler):
        assistant_indices = [
            i for i, item in enumerate(chat) if item['role'] == 'assistant'
        ]
        index_copy = index

        for i in assistant_indices:
            history = chat[:i]
            output = self.model.generate_from_template(
                [history], max_out_len=self.max_out_len)[0]
            candidate = extract_candidate_code(output)
            trimmed = trim_to_parsable_suffix_safe(candidate)
            chat[i]['content'] = trimmed
            if not self.dialogue_mode:
                output_handler.save_multiround_results(
                    origin_prompt=history[-1]['content'],
                    prediction=output,
                    idx=index,
                    gold=chat[i]['content'],
                )
        if self.dialogue_mode:
            assert len(chat) % 2 == 0
            round_num = int(len(chat) / 2)
            preds_list = []
            for i in range(round_num):
                temp_dict = {
                    'round': i + 1,
                    'user': chat[i * 2]['content'],
                    'assistant': chat[i * 2 + 1]['content']
                }
                preds_list.append(temp_dict)
            output_handler.save_results(
                origin_prompt=None,
                prediction=preds_list,
                idx=index_copy,
                gold=None,
            )
