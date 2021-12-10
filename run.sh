
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/0/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-6 --multi_label --do_eval

CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/3/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-9 --multi_label --do_eval

CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge-continue/p1-i0/4/p1-i0 --output_dir /raid/xiaoyuz1/goemotions_result/both-merge-epoch-10 --multi_label --do_eval


# Both Merged


CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/both-merge/p1-i0/4/p1-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/both-merge-continue --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 2 


CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/both-merge --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 2 


# Ekman 3 --> GoEmotions 27+1
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined_neg/p0-i0/2/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-ekman_neg --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 32 

CUDA_VISIBLE_DEVICES=4 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined_neg/p0-i0/2/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-ekman_neg-small-batch --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 2 



# Ekman 3
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined_neg --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/combined_neg --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 32


# Ekman 6+1 continue training the first Combined
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/combined-continue --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 32

# [dec 7th]
# Combine 27+1 GoEmotions and 6+1 Ekman (batch_size = 2)
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name both-merge --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/both_merge_6 --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/both_merge_6 --multi_label --do_train --pet_num_train_epochs 10 --pet_per_gpu_train_batch_size 2 


# 27+1 GoEmotions (batch_size = 2) on 6+1 Ekman (batch_size = 32)
CUDA_VISIBLE_DEVICES=7 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions_models/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/combined-continue-small-batch --multi_label --do_train --pet_num_train_epochs 10 --pet_per_gpu_train_batch_size 2 


# 6+1 Ekman (batch_size = 2)
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/combined-very-small-batch --multi_label --do_train --pet_num_train_epochs 10 --pet_per_gpu_train_batch_size 2



# Raw all the way
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1/p1-i0/1/p1-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1-continue --multi_label --do_train --pet_num_train_epochs 5 --pet_per_gpu_train_batch_size 32 --overwrite_output_dir
## train
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1/p1-i0/0/p1-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1 --multi_label --do_train --pet_num_train_epochs 2 --pet_per_gpu_train_batch_size 32 --overwrite_output_dir

## evaluate
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1/p1-i0/0/p1-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1/p1-i0/0 --multi_label --do_eval --pet_num_train_epochs 2 --pet_per_gpu_train_batch_size 32


## train
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1/p1-i0/0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1 --multi_label --do_train --pet_num_train_epochs 2 --pet_per_gpu_train_batch_size 32 --overwrite_output_dir

# DEBUG
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/DEBUG --multi_label --do_train --pet_num_train_epochs 2 --pet_per_gpu_train_batch_size 32



# Raw 
CUDA_VISIBLE_DEVICES=6 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-no-pretrain-pattern-1 --multi_label --do_train --pet_num_train_epochs 1 --pet_per_gpu_train_batch_size 32


# with DailyDialog small learning rate
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 1 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-pattern-1 --multi_label --do_train --pet_num_train_epochs 1 --pet_per_gpu_train_batch_size 32


CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined-small-batch/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-combined-small-batch --multi_label --do_train --pet_num_train_epochs 1 --pet_per_gpu_train_batch_size 32
--do_eval 

CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined --model_type bert --model_name bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/combined-small-batch --do_eval --multi_label --pet_num_train_epochs 1 --do_train 




# Goemotions
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name goemotions-prompt --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/data --model_type bert --model_name_or_path /raid/xiaoyuz1/goemotions/prompt/combined/p0-i0 --output_dir /raid/xiaoyuz1/goemotions/prompt/goemotion-prompt-small-batch --multi_label --do_train 

# without DailyDialog
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined --model_type bert --model_name bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/combined2 --pet_per_gpu_train_batch_size 32 --multi_label --do_train 

# with DailyDialog
CUDA_VISIBLE_DEVICES=5 python3 cli.py --method single --task_name combined --pattern_ids 0 --data_dir /raid/xiaoyuz1/goemotions/goemotions/combined --model_type bert --model_name bert-base-uncased --output_dir /raid/xiaoyuz1/goemotions/prompt/combined --pet_per_gpu_train_batch_size 32 --do_eval --multi_label

--do_train 

##
CUDA_VISIBLE_DEVICES=4 python3 cli.py \
--method single \
--pattern_ids 0 \
--data_dir /raid/xiaoyuz1/goemotions/goemotions/data/ekman/anger \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name ekman \
--output_dir /raid/xiaoyuz1/goemotions/prompt/anger \
--do_train \
--do_eval \
--pet_per_gpu_train_batch_size 32


# [multi_alt] [16 intensities + goemotions]
python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--aux_datasets /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/EI-oc/EIoc_goemotions_train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model multi_alt \
--epochs 30 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--training_seed 12345 \
--save_path /raid/xiaoyuz1/goemotions/save_path/multi_alt_semeval_oc/bert_seed-12345 \
--out_path /raid/xiaoyuz1/goemotions/pred_result/multi_alt_semeval_oc/test_pred_seed-12345.txt

python main.py predict \
--model_path /raid/xiaoyuz1/goemotions/save_path/multi_alt_semeval_oc/bert_seed-12345 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/sentiment/test_pred_debug.txt \
--out_label_path /raid/xiaoyuz1/goemotions/pred_result/multi_alt_semeval_oc/test_pred_seed-12345_label.pkl


# [has not tried yet, label number mismatch] direct fine tune? single-task [pretrained_on_semeval_ec]
python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--bert /raid/xiaoyuz1/goemotions/sem_eval_save_path/bert_seed-30010 \
--epochs 15 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--training_seed 12345 \
--save_path /raid/xiaoyuz1/goemotions/save_path/pretrained_on_semeval_ec/bert_seed-12345 \
--out_path /raid/xiaoyuz1/goemotions/pred_result/pretrained_on_semeval_ec/test_pred_seed-12345.txt

# sem eval 2018 e-c
python main.py train --main_dataset /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_train.csv --dev_file /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_dev.csv --test_file /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_test.csv --model bert --epochs 10 --lr 0.00001 --data /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_test.csv --training_seed 30010 --save_path /raid/xiaoyuz1/goemotions/sem_eval_save_path/bert_seed-30010 --out_path /raid/xiaoyuz1/goemotions/sem_eval_pred_result/test_pred_seed-30010.txt

for seed in 30010
do
   template="python main.py train --main_dataset /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_train.csv --dev_file /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_dev.csv --test_file /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_test.csv --model bert --epochs 20 --lr 0.00001 --data /raid/xiaoyuz1/goemotions/SemEval2018-Task1-all-data/English/E-c/goemotions_test.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/sem_eval_save_path/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/sem_eval_pred_result/test_pred_seed-${seed}.txt"
   echo $template
done

# ------------------------------------------------------------------------------------------------------------------------

# Sentiment
for seed in 30010 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train_sentiment.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev_sentiment.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test_sentiment.csv --model bert --epochs 6 --lr 0.00005 --data /raid/xiaoyuz1/goemotions/goemotions/data/test_sentiment.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/sentiment/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/sentiment/test_pred_seed-${seed}.txt"
   eval $template
   template2="python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/sentiment/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test_sentiment.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/sentiment/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/sentiment/test_pred_seed-${seed}_label.pkl"
   eval $template2
done

# FSJ
for seed in 30010 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train_fsj.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev_fsj.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test_fsj.csv --model bert --epochs 5 --lr 0.00005 --data /raid/xiaoyuz1/goemotions/goemotions/data/test_fsj.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/fsj/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/fsj/test_pred_seed-${seed}.txt"
   eval $template
   template2="python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/fsj/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test_fsj.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/fsj/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/fsj/test_pred_seed-${seed}_label.pkl"
   eval $template2
done

# Ekman
for seed in 79719 30010 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train_ekman.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev_ekman.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --model bert --epochs 5 --lr 0.00005 --data /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/ekman/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_seed-${seed}.txt"
   eval $template
   template2="python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/ekman/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_seed-${seed}_label.pkl"
   eval $template2
done
################################################################################################

python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--epochs 15 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--training_seed 12345 \
--save_path /raid/xiaoyuz1/goemotions/save_path/bert_paper_seed12345 \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_paper_seed12345.txt 

python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_paper_seed12345 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt \
--out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_paper_seed12345_label.pkl

################################################################################################
[x] 
python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--epochs 15 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--training_seed 12345 \
--save_path /raid/xiaoyuz1/goemotions/save_path/bert_paper_seed12345 \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_paper_seed12345.txt 


python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_paper_seed12345 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt \
--out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_paper_seed12345_label.pkl

################################################################################################


python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--epochs 15 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--training_seed 12345 \
--save_path /raid/xiaoyuz1/goemotions/save_path/bert_05_seed12345 \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_05_seed12345.txt 

python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_05_seed12345 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt \
--out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_05_seed12345_label.pkl


python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--save_path /raid/xiaoyuz1/goemotions/save_path/bert_03 \
--epochs 20 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_03.txt 

python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert --data /raid/xiaoyuz1/goemotions/goemotions/data/test_pred.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_label.pkl


python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_03 --data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_03_label.pkl

