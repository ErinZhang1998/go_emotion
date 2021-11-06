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

