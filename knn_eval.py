import os
import argparse
import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO


def compute_cider(nn_captions):
    # TODO: given a list of nearest neighbour captions, return nxn cider score
    # The order of captions should be preserved in the returned nxn matrix
    result = np.zeros((42))
    return result

def find_concensus_caption(nn_captions, m):
    '''
    Given a set of nearest neighbour captions, find the concensus caption
    '''
    
    # TODO: find cider scores between each caption
    # Assuming compute_cider returns a nxn matrix of cider score
    cider_scores = compute_cider(nn_captions)
    # Fill the diaganol to zeros
    np.fill_diagonal(cider_scores, 0.)
    # We can simply add all entries cause there's no negative cider score, so
    # always look for the max number of M
    # TODO: max over m 
    sum_of_scores = np.sum(cider_scores, axis=1)
    winner = np.argmax(sum_of_scores)
    return nn_captions[winner]    

def knn_concensus(args):
    # load knn matrix and coco files
    knn_mat = np.load(args.input_knn_mat)
    coco = json.load(args.coco_json)
    
    annFile = 'coco-caption/annotations/captions_val2014.json'
    coco_ann = COCO(annFile)
    ann_ids = list(coco_ann.anns.keys())
    coco_to_ann = defaultdict(list)
    for ann_id in ann_ids:
        coco_id = coco_ann.anns[ann_id]['image_id']
        coco_to_ann[coco_id].append(coco_ann.anns[ann_id]['caption'])

    coco_ids = np.load(args.info_npy).item().get('nondup_imgid')

    knn_to_coco = {}
    coco_to_knn = {}    
    for i, coco_id in enumerate(coco_ids):
        knn_to_coco[i] = coco_id
        coco_to_knn[coco_id] = i

    # Seperate out val and test images
    vals = []
    tests = []
    for ix in range(len(coco['images'])):
        img = coco['images'][ix]
        if img['split'] == 'val':
            vals.append(ix['id'])
        elif img['split'] == 'test':
            tests.append(ix['id'])

    results = []    
    # for each validation images
    for val_id in vals:
        # Get validation images' knn neighbour coco_id
        nn_ids = knn_mat[coco_to_knn[val_id]] 
        nn_coco_ids = [knn_to_coco[idx] for idx in nn_ids]
        nn_captions = []
        for nn_coco in nn_coco_ids:
            nn_captions += coco_to_ann[nn_coco]
        results.append({val_id: find_concensus_caption(nn_captions, args.m)})
    
    json.dump('knn_concensus.json', results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for evaluating knn \
                                                 scores using concensus')
    parser.add_argument('input_knn_mat', type=str,
                        default='data/knn_mat.npy')
    parser.add_argument('coco_json', type='str',
                        default='data/cocotalk.json')

    args = parser.parse_args()
    knn_concensus(args)
    # TODO: call eval_utils.py's language_eval(), replace preds
