import pandas as pd 

GO_EMOTIONS_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral',
]
GO_EMOTION_TEXT_TO_LABEL = dict(zip(
    GO_EMOTIONS_LABELS,
    range(len(GO_EMOTIONS_LABELS)),
))

def process_list_of_raw_labels(label_raw, target_num):
    label_text = []
    for l in label_raw:
        if int(l) == int(target_num):
            label_text.append(1)
    assert len(label_text) <= 1
    if len(label_text) > 0:
        return "1"
    else:
        return "0"

def read_goemotions(path, yes_emotion):
    '''
    :param path: .csv file with [text, label] columns for the GoEmotions dataset
    '''
    TABLE = {
        "anger" : 0, "disgust" : 1, "fear" : 2, "joy" : 3, "sadness" : 4, "surprise" : 5, "neutral" : 6,
    }
    
    df = pd.read_csv(path)
    texts = []
    labels = []

    for i in range(len(df)):
        row = df.iloc[i]
        texts.append(row['text'])
        label_raw = row['label'].strip().split(",")
        label = process_list_of_raw_labels(label_raw, TABLE[yes_emotion])
        labels.append(label)
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    
    return df


# AffectiveText
def read1(tweet_fs, emotion_fs, yes_emotion, thresh = 20):
    TABLE = {
        0 : "anger", 1 : "disgust", 2: "fear", 3: "joy", 4:"sadness", 5:"surprise", 6: "neutral",
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
                if yes_emotion != "neutral":
                    if int(l) >= thresh and TABLE[li] == yes_emotion:
                        label_text.append(1)
                else:
                    if int(l) >= thresh:
                        label_text.append(1)
                
            if yes_emotion != "neutral":
                assert len(label_text) <= 1
                if len(label_text) > 0:
                    labels.append("1")
                else:
                    labels.append("0")
            else:
                if len(label_text) == 0:
                    
                    labels.append("1")
                else:
                    labels.append("0")
            lineNo += 1
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    
    return df

def merge(paths):
    texts = []
    labels = []
    
    for path in paths:
        df = pd.read_csv(path)
        texts += list(df['text'].to_numpy())
        labels += list(df['label'].to_numpy())
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    
    return df