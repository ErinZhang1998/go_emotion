import pandas as pd 

# AffectiveText
def read1(tweet_fs, emotion_fs, yes_emotion, thresh = 20):
    TABLE = {
        0 : "anger", 1 : "disgust", 2: "fear", 3: "joy", 4:"sadness", 5:"surprise",
    }
    
    texts = []
    labels = []
    
    for tweet_f, emotion_f in zip(tweet_fs, emotion_fs):
    
        fh1 = open(tweet_f, "r")
        fh2 = open(emotion_f, "r")
        L1 = fh1.readlines()
        L2 = fh2.readlines()

        lineNo = 0
    
    
        for line in L1:
            if not line.startswith("<instance"):
                continue 
            text = line.split(">")[1].split("<")[0]
            label_raw = L2[lineNo].strip().split(" ")[1:]

            texts.append(text)
            label_text = []
            for li, l in enumerate(label_raw):
                if int(l) >= thresh and TABLE[li] == yes_emotion:
                    label_text.append(str(li))
            
            if len(label_text) > 0:
                labels.append(",".join(label_text))
            else:
                labels.append("6")
            lineNo += 1
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    
    return df

tweet_f1 = "/raid/xiaoyuz1/goemotions/metaData/AffectiveText.test/affectivetext_test.xml"
emotion_f1 = "/raid/xiaoyuz1/goemotions/metaData/AffectiveText.test/affectivetext_test.emotions.gold"
tweet_f2 = "/raid/xiaoyuz1/goemotions/metaData/AffectiveText.trial/affectivetext_trial.xml"
emotion_f2 = "/raid/xiaoyuz1/goemotions/metaData/AffectiveText.trial/affectivetext_trial.emotions.gold"

df1 = read1([tweet_f1, tweet_f2], [emotion_f1, emotion_f2])
# df1.to_csv()