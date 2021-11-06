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

