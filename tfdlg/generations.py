import numpy as np
from scipy.special import softmax


def filter_to_topk(top_k, dist):
    """Replace all the elements in dist to -Inf
    except the top k largest elements.
    Args:
        top_k (int):
        dist (torch.Tensor): (num_batch, -1) dimensioned torch tensor.
    Returns:
        torch.Tensor: (num_batch, -1) dimensioned torch tensor.
    """
    dist = np.copy(dist)

    batch_size = dist.shape[0]
    n_cands = dist.shape[1]
    top_k = min(top_k, n_cands)

    top_k_index = np.argsort(-dist)[:, :top_k]
    top_k_value = np.take_along_axis(dist, top_k_index, axis=-1)

    threshold = top_k_value[:, -1]  # dim = (batch_size, )
    threshold = threshold.reshape((batch_size, 1))  # dim = (batch_size, 1)

    dist[dist < threshold] = -np.inf
    return dist


def filter_to_topp(top_p, dist):
    """
    Args:
        top_k (int):
        dist (torch.Tensor): (num_batch, -1) dimensioned torch tensor.
    Returns:
        torch.Tensor: (num_batch, -1) dimensioned torch tensor.
    """
    dist = np.copy(dist)

    batch_size = dist.shape[0]

    sorted_index = np.argsort(-dist)
    sorted_value = np.take_along_axis(dist, sorted_index, axis=-1)

    sorted_prob = softmax(sorted_value, axis=1)
    sorted_prob_cumsum = np.cumsum(sorted_prob, axis=-1)

    # shift right side
    # Algorithm
    # cumsum = [0.1, 0.3, 0.93, 0.1] and top_p = 0.9
    # cumsum > top_p -> [False, False, True,  True]
    # shift          -> [False, False, False, True]
    sorted_index_mask = np.concatenate(
        [np.full((batch_size, 1), False),
         (sorted_prob_cumsum > top_p)[:, :-1]],
        axis=-1
    )

    mask = np.full(dist.shape, False)
    np.put_along_axis(mask, sorted_index, sorted_index_mask, axis=-1)

    dist[mask] = -float("Inf")

    return dist


def filter_bad_ids(bad_ids, dist):
    dist = np.copy(dist)
    dist[:, bad_ids] = -float("Inf")
    return dist


def sample_multinomial(dist):
    # dist should be casted to float64 before passing multinomial
    # because multinomial implicitly cast the inputs to float64.
    # If the type of result of softmax is float32 and converted to float64,
    # the sum of the softmax will be over 1.0.
    # https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
    dist = dist.astype(np.float64)

    # np.random.multinomial works only with one dimensional array
    spl_one_hot = np.array(
        [np.random.multinomial(n=1, pvals=softmax(dist_one)) for dist_one in dist],
        dtype=dist.dtype
    )  # shape: dist.shape
    spl_label = np.argmax(spl_one_hot, axis=1)  # shape: (dist.shape[1], )
    return spl_label


class TopKTopPGenerator:
    """Sentence generator to sandom sampling from top-k distribution"""
    def __init__(self, model, top_k=50, top_p=0.95,
                 bad_ids=[], max_len=20, stop_id=None):
        self._model = model
        self._top_p = top_p
        self._top_k = top_k
        self._bad_ids = bad_ids
        self._max_len = max_len
        self._stop_id = stop_id

    def step(self, inputs):
        """
        Args:
            inputs: shape == (batch_size, seq_len)
        """
        outputs = self._model(inputs)  # shape == (batch_size, seq_len, vocab_size)

        next_id_dist = outputs[:, -1, :]

        # Set filter_bad_ids first
        # If not, all values would be -inf, which leads to raise exception
        # when calculating softmax
        filters = [
            lambda dist: filter_bad_ids(self._bad_ids, dist),
            lambda dist: filter_to_topp(self._top_p, dist),
            lambda dist: filter_to_topk(self._top_k, dist),
        ]
        filtered_dist = next_id_dist
        for flt in filters:
            filtered_dist = flt(filtered_dist)

        ids = sample_multinomial(filtered_dist)

        logproba = - np.log(softmax(filtered_dist.astype(np.float64)))
        proba = np.array([logproba[i][idx] for i, idx in enumerate(ids)])

        return ids, proba

    def generate(self, inputs):
        need_stop = np.array([False for _ in range(inputs.shape[0])])
        probas = []

        for _ in range(self._max_len):
            outputs, one_probas = self.step(inputs)
            need_stop = need_stop | (outputs == self._stop_id)
            if self._stop_id is not None and np.all(need_stop):
                break

            outputs = outputs.reshape((inputs.shape[0], 1))
            inputs = np.concatenate([inputs, outputs], axis=-1)
            probas.append(one_probas)
        return inputs, np.array(probas).T
