SOURCES=(46921 25577 52538 56440 41228 66558 48642 69556) #79719 30010 46921 25577 52538 56440 41228 66558 48642 69556
for seed in 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="CUDA_VISIBLE_DEVICES=4,5 python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --model bert --epochs 20 --data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_seed-${seed}.txt"
   eval $template
   template2="CUDA_VISIBLE_DEVICES=4,5 python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_seed-${seed}_label.pkl"
   eval $template2
done