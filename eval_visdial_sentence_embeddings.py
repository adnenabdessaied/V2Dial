from sentence_transformers.cross_encoder import CrossEncoder
import os
import torch
import json
import numpy as np 

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


class SparseGTMetrics(object):
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self):
        self._rank_list = []

    def observe(
        self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()

        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]
        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

    def retrieve(self, reset: bool = True):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                "r@1": torch.mean((__rank_list <= 1).float()).item(),
                "r@5": torch.mean((__rank_list <= 5).float()).item(),
                "r@10": torch.mean((__rank_list <= 10).float()).item(),
                "mean": torch.mean(__rank_list).item(),
                "mrr": torch.mean(__rank_list.reciprocal()).item(),
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._rank_list = []


class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(1)
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
               
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0


annos_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/annotations/visdial_v1.0/visdial_1.0_val.json'
with open(annos_path, 'r') as f:
    data = json.load(f)['data']

dense_annos_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/annotations/visdial_v1.0/visdial_1.0_val_dense_annotations.json'
with open(dense_annos_path, 'r') as f:
    dense_data = json.load(f)

dense_data = {str(d['image_id']) + '_' + str(d['round_id']): d['gt_relevance'] for d in dense_data}

results_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/output/visdial_before_supplementary/zeroshot_visdial_after_avsd_4_frames_3_rounds_ft_fp16_googleflant5large_results_dstc10_beam_depth_4_lenPen_0.3.json'
with open(results_path, 'r') as f:
    results = json.load(f)

all_answers   = data['answers']
all_questions = data['questions']


dialogs = data['dialogs']

dialogs_dict = {}

for dialog in dialogs:
    image_id = dialog['image_id']
    for i, turn in enumerate(dialog['dialog']):
        answer_opts = [all_answers[a] for a in turn['answer_options']]
        dialogs_dict[str(image_id) + '_' + str(i+1)] = {
            'answer_opts': answer_opts,
            'gt_index': turn['gt_index'] 
        }
        # print('bla')

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/stsb-roberta-large")

for i, (res_key, res) in enumerate(results.items()):
    print('[INFO] {} / {}'.format(i+1, len(results)))
    answer_opts = dialogs_dict[res_key]['answer_opts']
    gt_index = torch.tensor(dialogs_dict[res_key]['gt_index'])
    gt_answer = answer_opts[gt_index]
    sentence_combinations = [[res, opt] for opt in answer_opts]
    scores = model.predict(sentence_combinations)
    scores = torch.from_numpy(scores).unsqueeze(0).unsqueeze(0)
    # scores = torch.tensor([ratio(res, answer_opt) for answer_opt in answer_opts]).unsqueeze(0).unsqueeze(0)
    # scores = model.rank(res, answer_opts)
    ranked_idx = scores_to_ranks(scores).squeeze().tolist()
    new_order = np.argsort(ranked_idx)
    # ranked_answers = [answer_opts[idx] for idx in new_order]
    best_pick = answer_opts[new_order[0]]
    sparse_metrics.observe(scores, gt_index)
    if res_key in dense_data:
        gt_relevance = torch.tensor(dense_data[res_key]).unsqueeze(0)
        ndcg.observe(scores.squeeze(0), gt_relevance)

    # print('bla')
print(sparse_metrics.retrieve())
print(ndcg.retrieve())

# We want to compute the similarity between the query sentence...
# query = "A man is eating pasta."

# # ... and all sentences in the corpus
# corpus = [
#     "A man is eating food.",
#     "A man is eating a piece of bread.",
#     "The girl is carrying a baby.",
#     "A man is riding a horse.",
#     "A woman is playing violin.",
#     "Two men pushed carts through the woods.",
#     "A man is riding a white horse on an enclosed ground.",
#     "A monkey is playing drums.",
#     "A cheetah is running behind its prey.",
# ]

# # 2. We rank all sentences in the corpus for the query
# ranks = model.rank(query, corpus)

# # Print the scores
# print("Query: ", query)
# for rank in ranks:
#     print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")
# """
# Query:  A man is eating pasta.
# 0.67    A man is eating food.
# 0.34    A man is eating a piece of bread.
# 0.08    A man is riding a horse.
# 0.07    A man is riding a white horse on an enclosed ground.
# 0.01    The girl is carrying a baby.
# 0.01    Two men pushed carts through the woods.
# 0.01    A monkey is playing drums.
# 0.01    A woman is playing violin.
# 0.01    A cheetah is running behind its prey.
# """

# # 3. Alternatively, you can also manually compute the score between two sentences
# import numpy as np

# sentence_combinations = [[query, sentence] for sentence in corpus]
# scores = model.predict(sentence_combinations)

# # Sort the scores in decreasing order to get the corpus indices
# ranked_indices = np.argsort(scores)[::-1]
# print("Scores:", scores)
# print("Indices:", ranked_indices)
# """
# Scores: [0.6732372, 0.34102544, 0.00542465, 0.07569341, 0.00525378, 0.00536814, 0.06676237, 0.00534825, 0.00516717]
# Indices: [0 1 3 6 2 5 7 4 8]
# """