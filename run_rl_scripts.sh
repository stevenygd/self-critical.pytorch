#! /bin/bash

# python train.py --id log_test_rl --caption_model topdown --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/cocotalk_label.h5 --learning_rate 0.01 --start_from log_test_rl --checkpoint_path beam_log_lr --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 61 --start_from log_test_rl --beam_size 5 --input_encoding_size 1000 --rnn_size 1000 --batch_size 100 --optim sgdmom --max_epoch 60 --weight_decay 5e-4

python train.py --id log_test_rl --caption_model topdown --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/cocotalk_label.h5 --learning_rate 0.01 --checkpoint_path sup_repro_813 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 60 --beam_size 5 --input_encoding_size 1000 --rnn_size 1000 --batch_size 100 --optim sgdmom --max_epoch 60 --weight_decay 5e-4
