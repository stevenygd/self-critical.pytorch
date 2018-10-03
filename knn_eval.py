import os
import argparse
import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from itertools import combinations
from eval_utils import language_eval

def compute_cider(nn_captions, scorer):
    # Given a list of nearest neighbour captions, return nxn cider score
    # The order of captions should be preserved in the returned nxn matrix
    num_cand = len(nn_captions)
    result = np.zeros((num_cand, num_cand))
    gts = {}
    res = []
    i = 0
    for gt, re in combinations(nn_captions, 2):
        gts[i] = [gt]
        res.append({'image_id':i, 'caption':[re]})
        i += 1
    _, scores = scorer.compute_score(gts, res)
    # filling up the upper and lower triangle
    iu = np.triu_indices(num_cand, 1)
    result[iu] = scores
    # fill the lower half by adding lower half
    result = result + result.T
    return result


def find_concensus_caption(nn_captions, m, scorer):
    '''
    Given a set of nearest neighbour captions, find the concensus caption
    '''

    # Assuming compute_cider returns a nxn matrix of cider score
    cider_scores = compute_cider(nn_captions, scorer)
    # make sure the diaganol are zeros
    np.fill_diagonal(cider_scores, 0.)
    # Find the max for m entries, given we don't have negative scores
    sorted_scores = np.sort(cider_scores, axis=1)
    top_m_sum = np.sum(sorted_scores[:, -m:], axis=1)
    winner = np.argsort(top_m_sum)[-1]
    return nn_captions[winner]


def knn_concensus(args):
    # load knn matrix and coco files
    knn_mat = np.load(args.input_knn_mat)
    coco = json.load(open(args.coco_json))

    # annFile = 'coco-caption/annotations/captions_train2014.json'
    # coco_ann = COCO(annFile)
    # ann_ids = list(coco_ann.anns.keys())
    coco_to_ann = defaultdict(list)
    # for ann_id in ann_ids:
    #     coco_id = coco_ann.anns[ann_id]['image_id']
    #     coco_to_ann[coco_id].append(coco_ann.anns[ann_id]['caption'])

    annFile = 'data/dataset_coco.json'
    coco_ann = json.load(open(annFile))
    # coco_ann = COCO(annFile)
    # ann_ids = list(coco_ann.anns.keys())
    # for ann_id in ann_ids:
    #     coco_id = coco_ann.anns[ann_id]['image_id']
    #     coco_to_ann[coco_id].append(coco_ann.anns[ann_id]['caption'])
    # print('finish loading coco annotation')
    for coco_item in coco_ann['images']:
        for sent in coco_item['sentences']:
            coco_to_ann[coco_item['cocoid']].append(sent['raw'])
    print('finish setting up coco to annotation')

    coco_ids = np.load(args.info_npy).item().get('nondup_imgid')

    knn_to_coco = {}
    coco_to_knn = {}
    for i, coco_id in enumerate(coco_ids):
        knn_to_coco[i] = coco_id
        coco_to_knn[coco_id] = i
    print('finished setting up knn to coco lookup')

    # Seperate out val and test images, don't do anything to test yet
    vals = []
    tests = []
    for ix in range(len(coco['images'])):
        img = coco['images'][ix]
        if img['split'] == 'val':
            vals.append(img['id'])
        elif img['split'] == 'test':
            tests.append(img['id'])

    # Set up cider scorer
    from pyciderevalcap.cider.cider import Cider
    scorer = Cider(df='coco-val')
    print('created cider scorer')

    print('starting to eval concensus knn......')
    from progressbar import progressbar
    results = []
    # for each validation images
    for val_id in progressbar(vals):
        # Get validation images' knn neighbour coco_id
        nn_ids = knn_mat[coco_to_knn[val_id]]
        nn_coco_ids = [knn_to_coco[idx] for idx in nn_ids]
        nn_captions = []
        for nn_coco in nn_coco_ids:
            nn_captions += coco_to_ann[nn_coco]
        results.append({'image_id':val_id, 'caption':find_concensus_caption(nn_captions, args.m, scorer)})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for evaluating knn \
                                                 scores using concensus')
    parser.add_argument('--input_knn_mat', type=str,
                        default='data/resnet-features-coco2014/peteranderson-fcfeat-knnidx.npy')
    parser.add_argument('--coco_json', type=str,
                        default='data/cocotalk.json')
    parser.add_argument('--info_npy', type=str,
                        default='data/cocobu_trainval_nn/info.npy')
    parser.add_argument('--m', type=int, default=5)

    args = parser.parse_args()
    results = knn_concensus(args)
    language_eval(None, results, 'knn_eval', 'val')
    json.dump(results, open('knn_concensus.json', 'w'))
