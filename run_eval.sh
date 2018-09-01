#! /bin/bash

ID="sup_repro_820_adam"
python eval_all.py \
    --id ${ID} \
    --caption_model topdown \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocobu_fc \
    --input_att_dir data/cocobu_att \
    --input_label_h5 data/cocotalk_label.h5 \
    --checkpoint_path ${ID} \
    --save_checkpoint_every 2000 \
    --language_eval 1 \
    --val_images_use 5000 \
    --self_critical_after 60 \
    --input_encoding_size 1000 \
    --rnn_size 1000 \
    --batch_size 100 \
    --max_epoch 100 \
    --learning_rate 5e-4 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --learning_rate_decay_every 6 \
    --scheduled_sampling_increase_every 10 \
    --beam_size 5 \
    --start_from ${ID}
