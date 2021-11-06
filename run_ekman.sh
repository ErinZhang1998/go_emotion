SOURCES=(79719 30010 46921 25577 52538 56440 41228 66558 48642 69556)
for seed in 79719 30010 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train_ekman.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev_ekman.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --model bert --epochs 20 --data /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/ekman/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_seed-${seed}.txt"
   eval $template
   template2="python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/ekman/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test_ekman.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/ekman/test_pred_seed-${seed}_label.pkl"
   eval $template2
done