import csv
import pickle
from main import train_setup
from tqdm import tqdm


from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

with open("/raid/xiaoyuz1/goemotions/save_path/bert-meta.pkl", "rb") as f:
    meta = pickle.load(f)

fn = '/raid/xiaoyuz1/goemotions/goemotions/data/test_pred.csv'
with open(fn, "r", encoding="utf-8") as f:
    data = list(csv.reader(f))[1:]  # for debug, do [1:1000]

results = []

for i in tqdm(range(len(data))):
    text, label = data[i]
    emotion_labels = emotion(text)
    results.append(emotion_labels)

with open('/raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_hugging.pkl', "wb+") as f:
    pickle.dump(results, f)

# all_preds = []
# all_golds = []
