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

def read2(paths, yes_emotion, thresh = 0.2):
    texts = []
    labels = []
    for path in paths:
        fh = open(path, "r")
        L1 = fh.readlines()
        for line in L1:
            parts = line.split("\t")
            text = parts[1].strip()
            texts.append(text)
            label = parts[2].strip()
            intensity = float(parts[3].strip())
            
            if yes_emotion != "neutral":

                if(intensity >= thresh) and label == yes_emotion:
                    labels.append("1")
                else:
                    labels.append("0")
            else:
                if(intensity < thresh):
                    labels.append("1")
                else:
                    labels.append("0")
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)

    
    return df 

def read3(paths, yes_emotion):
    texts = []
    labels = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        for i in range(len(df)):
            row = df.iloc[i]
            texts.append(row["Tweet"])

            if yes_emotion != "neutral":
                if yes_emotion in row:
                    if str(row[yes_emotion]) == "1":
                        labels.append("1")
                        continue 
                labels.append("0")
            else:
                all_not_set = True 
                for k,v in row.items():
                    if str(v) == "1" or str(v) == "0":
                        if str(v) == "1":
                            all_not_set = False 
                if all_not_set:
                    labels.append("1")
                else:
                    labels.append("0")
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)
    return df 

def read4(paths, yes_emotion, thresh=0.2):
    texts = []
    labels = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        for i in range(len(df)):
            row = df.iloc[i]
            texts.append(row["Tweet"])

            if yes_emotion != "neutral":
                if row["Affect Dimension"] == yes_emotion:
                    if row["Intensity Score"] >= thresh:
                        labels.append("1")
                        continue
                labels.append("0")
            else:
                if row["Intensity Score"] < thresh:
                    labels.append("1")
                else:
                    labels.append("0")
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)
    return df 

def read5(text_file, emotion_file, yes_emotion):
    TABLE = { 0: "neutral", 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'sadness', 6: 'surprise'}
    fh1 = open(text_file, "r")
    L1 = fh1.readlines()

    fh2 = open(emotion_file, "r")
    L2 = fh2.readlines()

    texts = []
    labels = []

    assert len(L1) == len(L2)
    for line,emo in zip(L1,L2):
        parts = line.strip().split("__eou__")
        emos = emo.strip().split(" ")

        # print(line, parts, emos)

        # print(len(parts),len(emos))
        # assert len(parts) == len(emos)
        
        for part, l in zip(parts, emos):
            texts.append(part.strip())
            l = int(l)
            if TABLE[l] == yes_emotion:
                labels.append("1")
            else:
                labels.append("0")
    
    df = pd.DataFrame({'text': texts,
                   'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)
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