from typing import List


def encode_dialog(encode_fn, sep_token_id, context: List[str], response: str=None):
    ids = []
    for i, txt in enumerate(context):
        ids += [sep_token_id]
        ids += encode_fn(txt)
    ids += [sep_token_id]

    if response is not None:
        ids += encode_fn(response)

    # return np.array(ids, dtype=np.int32)
    return ids
