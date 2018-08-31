from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
           'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
           'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
           'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']

# For testing
# infiles = ['test2014_resnet101_faster_rcnn_genome_36.tsv']


os.makedirs(args.output_dir+'_att')
os.makedirs(args.output_dir+'_fc')
os.makedirs(args.output_dir+'_box')
os.makedirs(args.output_dir+'_nn')

imgid = []
nondup_imgid = []
chipid = [] # str(imgid + '_' + ctr)
feats = []
fc_feats = []
for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        # reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        # for item in reader:
        for line in tsv_in_file:
            lineitem = line.strip().split(b'\t', len(FIELDNAMES))
            item = {}
            item['image_id'] = int(lineitem[FIELDNAMES.index('image_id')])
            item['num_boxes'] = int(lineitem[FIELDNAMES.index('num_boxes')])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(lineitem[FIELDNAMES.index(field)]),
                        dtype=np.float32).reshape((item['num_boxes'],-1))
            for ctr in range(item['num_boxes']):
                imgid.append(item['image_id'])
                chipid.append("%d_%d"%(item['image_id'], ctr))
            nondup_imgid.append(item['image_id'])
            feats.append(item['features'])
            fc_feats.append(item['features'].mean(0))
            np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features']) # 32x2048
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])),
                item['features'].mean(0)) # 32x2048 -> 2048
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])

feats = np.concatenate(feats, axis=0)
fc_feats = np.stack(fc_feats, axis=0)
np.savez_compressed(os.path.join(args.output_dir+"_nn", "feat.npz"), feats)
np.savez_compressed(os.path.join(args.output_dir+"_nn", "fc_feat.npz"), fc_feats)
infos = {
        'imgid' : imgid,
        'chipid' : chipid,
        'nondup_imgid' :nondup_imgid
        }
np.save(os.path.join(args.output_dir+"_nn", "info.npy"), infos)


