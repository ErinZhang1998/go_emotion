# 
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/4/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-10 --multi_label --do_eval
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/3/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-9 --multi_label --do_eval
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/0/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-6 --multi_label --do_eval

CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-ekman_neg-small-batch/p1-i0/4/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/goemotion-prompt-ekman_neg-small-batch-epoch-5 --multi_label --do_eval

CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1-continue/p1-i0/4/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/goemotion-prompt-no-pretrain-10 --multi_label --do_eval


CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge_6 --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both_merge_6/p1-i0/6/p1-i0/ --output_dir /raid/xiaoyuz1/goemotions_result/both_merge_6-epoch-6 --multi_label --do_eval

CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined-continue-small-batch/p1-i0/8/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/6-separate --multi_label --do_eval
