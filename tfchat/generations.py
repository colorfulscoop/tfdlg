import numpy as np


def filter_to_topk(top_k, dist):
    """Replace all the elements in dist to -Inf
    except the top k largest elements.
    Args:
        top_k (int):
        dist (torch.Tensor): (num_batch, -1) dimensioned torch tensor.
    Returns:
        torch.Tensor: (num_batch, -1) dimensioned torch tensor.
    """
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
    dist = dist.clone()

    # sort for each row
    sorted_dist, sorted_idx = torch.sort(dist, descending=True)
    # cumulative probability sum for each row
    prob_dist_cusum = torch.cumsum(
        torch.nn.functional.softmax(sorted_dist, dim=-1),
        dim=-1
    )

    # Detect the filter index
    removed_index = torch.cat(
        [
            torch.tensor([[False] for _ in range(prob_dist_cusum.shape[0])]),
            prob_dist_cusum > top_p,
        ],
        dim=1
    )[:, :-1]

    # pass (dim, index, source)
    mask_flag = removed_index.scatter(1, sorted_idx, removed_index)
    dist[mask_flag] = -float("Inf")

    return dist


def filter_bad_ids(bad_ids, dist):
    dist = dist.clone()
    dist[:, bad_ids] = -float("Inf")
    return dist


def sample_multinomial(dist):
    return torch.multinomial(
        input=torch.functional.F.softmax(dist, dim=-1),
        num_samples=1
    )


class TopPKGenerator:
    """Sentence generator to sandom sampling from top-k distribution"""
    def __init__(self, model, top_p, top_k, bad_ids):
        self._model = model
        self._top_p = top_p
        self._top_k = top_k
        self._bad_ids = bad_ids

    def step(self, **argv):
        # Predict next word distribution
        output = self._model(**argv)
        # last_hidden_state dim = (batch_size, input_ids length, num_vocabs)
        last_hidden_state = output[0]
        next_id_dist = last_hidden_state[:, -1, :]

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

        return sample_multinomial(filtered_dist), output