#! /bin/bash

# python train.py --id log_test_rl --caption_model topdown --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/cocotalk_label.h5 --learning_rate 0.01 --start_from log_test_rl --checkpoint_path beam_log_lr --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 61 --start_from log_test_rl --beam_size 5 --input_encoding_size 1000 --rnn_size 1000 --batch_size 100 --optim sgdmom --max_epoch 60 --weight_decay 5e-4

# # From scratch
# ID="sup_repro_814"
# python train.py \
#     --id ${ID} \
#     --caption_model topdown \
#     --input_json data/cocotalk.json \
#     --input_fc_dir data/cocobu_fc \
#     --input_att_dir data/cocobu_att \
#     --input_label_h5 data/cocotalk_label.h5 \
#     --learning_rate 0.01 \
#     --checkpoint_path ${ID} \
#     --save_checkpoint_every 2000 \
#     --language_eval 1 \
#     --val_images_use 5000 \
#     --self_critical_after 60 \
#     --beam_size 5 \
#     --input_encoding_size 1000 \
#     --rnn_size 1000 \
#     --batch_size 100 \
#     --optim sgdmom \
#     --max_epoch 60 \
#     --weight_decay 5e-4

# # From scratch
# ID="sup_repro_814_adam"
# python train.py \
#     --id ${ID} \
#     --caption_model topdown \
#     --input_json data/cocotalk.json \
#     --input_fc_dir data/cocobu_fc \
#     --input_att_dir data/cocobu_att \
#     --input_label_h5 data/cocotalk_label.h5 \
#     --checkpoint_path ${ID} \
#     --save_checkpoint_every 2000 \
#     --language_eval 1 \
#     --val_images_use 5000 \
#     --self_critical_after 60 \
#     --beam_size 5 \
#     --input_encoding_size 1000 \
#     --rnn_size 1000 \
#     --batch_size 100 \
#     --max_epoch 60 \
#     --weight_decay 5e-4 \
#     --learning_rate 5e-4 \
#     --learning_rate_decay_start 0 \
#     --scheduled_sampling_start 0
#     # --optim sgdmom \
#     # --learning_rate 0.01 \


# # From scratch : CIDEr 113
# ID="sup_repro_819_adam"
# python train.py \
#     --id ${ID} \
#     --caption_model topdown \
#     --input_json data/cocotalk.json \
#     --input_fc_dir data/cocobu_fc \
#     --input_att_dir data/cocobu_att \
#     --input_label_h5 data/cocotalk_label.h5 \
#     --checkpoint_path ${ID} \
#     --save_checkpoint_every 2000 \
#     --language_eval 1 \
#     --val_images_use 5000 \
#     --self_critical_after 60 \
#     --input_encoding_size 1000 \
#     --rnn_size 1000 \
#     --batch_size 100 \
#     --max_epoch 60 \
#     --learning_rate 5e-4 \
#     --learning_rate_decay_start 0 \
#     --scheduled_sampling_start 0 \
#     --beam_size 5;


# ID="sup_repro_819_adam_0.9rate"
# python train.py \
#     --id ${ID} \
#     --caption_model topdown \
#     --input_json data/cocotalk.json \
#     --input_fc_dir data/cocobu_fc \
#     --input_att_dir data/cocobu_att \
#     --input_label_h5 data/cocotalk_label.h5 \
#     --checkpoint_path ${ID} \
#     --save_checkpoint_every 2000 \
#     --language_eval 1 \
#     --val_images_use 5000 \
#     --self_critical_after 60 \
#     --input_encoding_size 1000 \
#     --rnn_size 1000 \
#     --batch_size 100 \
#     --max_epoch 60 \
#     --learning_rate 5e-4 \
#     --learning_rate_decay_start 0 \
#     --scheduled_sampling_start 0 \
#     --learning_rate_decay_rate 0.9 \
#     --beam_size 5;

# ID="sup_repro_820_adam"
# python train.py \
#     --id ${ID} \
#     --caption_model topdown \
#     --input_json data/cocotalk.json \
#     --input_fc_dir data/cocobu_fc \
#     --input_att_dir data/cocobu_att \
#     --input_label_h5 data/cocotalk_label.h5 \
#     --checkpoint_path ${ID} \
#     --save_checkpoint_every 2000 \
#     --language_eval 1 \
#     --val_images_use 5000 \
#     --self_critical_after 60 \
#     --input_encoding_size 1000 \
#     --rnn_size 1000 \
#     --batch_size 100 \
#     --max_epoch 60 \
#     --learning_rate 5e-4 \
#     --learning_rate_decay_start 0 \
#     --scheduled_sampling_start 0 \
#     --learning_rate_decay_every 6 \
#     --scheduled_sampling_increase_every 10 \
#     --beam_size 5;

# CIDEr Optimization : change the max epochs
ID="sup_repro_820_adam"
python train.py \
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
    --start_from ${ID};
