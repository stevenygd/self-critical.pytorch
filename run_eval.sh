#! /bin/bash

ID="cross_entropy_nnedit"
python eval_all.py \
    --id ${ID} \
    --caption_model topdownnnedit \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocobu_fc \
    --input_att_dir data/cocobu_att \
    --input_nn_dir data/cocobu_nn \
    --input_box_dir data/cocobu_box \
    --input_knn_mat data/resnet-features-coco2014/peteranderson-fcfeat-knnidx.npy \
    --input_label_h5 data/cocotalk_label.h5 \
    --checkpoint_path ${ID} \
    --save_checkpoint_every 1000 \
    --language_eval 1 \
    --val_images_use 5000 \
    --input_encoding_size 1000 \
    --rnn_size 1000 \
    --batch_size 64 \
    --max_epoch 60 \
    --learning_rate 5e-4 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --learning_rate_decay_every 6 \
    --scheduled_sampling_increase_every 10 \
    --beam_size 5 \
