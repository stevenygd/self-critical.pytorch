#! /bin/bash

python train.py --id test --caption_model topdown --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc/ --input_att_dir data/cocobu_att/ --input_box_dir data/cocobu_box/ --input_label_h5 data/cocotalk_label.h5 --batch_size=32 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_test --save_checkpoint_every 10 --val_images_use 50 --max_epochs 30 --language_eval 1
