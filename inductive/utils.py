import numpy as np
from scipy.stats import rankdata

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True)
    full_rank = rankdata(-scores, method='ordinal', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='ordinal', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10


def cal_dcg(rel):
    dcg = np.sum((np.power(2, rel)-1) / np.log2(np.arange(len(rel), dtype=np.float32)+2))
    return dcg


def cal_ndcg(scores, labels, filters, n=10):
    # ndcg topk
    denom = np.log2(np.arange(2, n + 2))
    ndcgs = []
    p_topks = []
    r_topks = []
    for i in range(len(scores)):
        scores_i = scores[i]
        sorted_list_tmp = np.argsort(scores_i, axis=0, kind='stable')[::-1]
        gt = np.nonzero(labels[i])[0]

        hit_topk = len(np.intersect1d(sorted_list_tmp[:n], gt))

        dcg_topk = np.sum(np.in1d(sorted_list_tmp[:n], gt) / denom)
        idcg_topk = np.sum((1 / denom)[:min(len(gt), n)])
        ndcg = dcg_topk / idcg_topk if idcg_topk != 0 else 0
        p_topk = hit_topk / min(len(gt), n) if len(gt) != 0 else 0
        r_topk = hit_topk / len(gt) if len(gt) != 0 else 0
        ndcgs.append(ndcg)
        p_topks.append(p_topk)
        r_topks.append(r_topk)
    return np.mean(ndcgs), np.mean(p_topks), np.mean(r_topks)
