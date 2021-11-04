python main.py train \
--main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv \
--dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv \
--test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv \
--model bert \
--save_path /raid/xiaoyuz1/goemotions/save_path/bert \
--epochs 20 \
--data /raid/xiaoyuz1/goemotions/goemotions/data/test_pred.csv \
--out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred.txt 

python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert --data /raid/xiaoyuz1/goemotions/goemotions/data/test_pred.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_label.pkl
