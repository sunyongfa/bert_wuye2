import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

'''i = 0
    id2label = {}
    label2id = {}
    with open("id2label.txt", 'w', encoding="utf-8") as f1, open("label2id.txt", 'w', encoding="utf-8") as f2:
        for line in set(data[0]):
            id2label[str(i)] = line.strip()
            label2id[line.strip()] = str(i)
            i += 1
        f1.write(str(id2label))
        f2.write(str(label2id))'''

with open('data/label2id.txt', 'r', encoding="utf-8") as f:
    label2id0 = eval(f.read())
label2id1 = {"正面": 0, "负面": 1, "闲聊": 2, "中性": 3}
label2idlist=[label2id0,label2id1]

file0="data/warning"
file1="data/sentiment"
file0list=["all/train.tsv","test1/train.tsv","test1/dev.tsv","test1/test.tsv","test2/train.tsv","test2/dev.tsv","test2/test.tsv"]
file1list=["all/train.tsv","test1/train.tsv","test1/dev.tsv","test1/test.tsv","test2/train.tsv","test2/dev.tsv","test2/test.tsv"]
file0lists=[os.path.join(file0, f) for f in file0list]
file1lists=[os.path.join(file1, f) for f in file1list]
filelists=[file0lists,file1lists]


def writetofile(file,data,label2id):
    with open(file, 'w', encoding="utf-8") as f:
        for i in range((len(data[0]))):
            line = str(data[1].iloc[i]).replace("\t", "").replace("\n", "").replace("\r", "").replace(" ",
                                                                                                      "").strip()
            s = str(data[0].iloc[i]).strip()
            f.write(str(label2id[s]) + "\t" + line + "\n")


def splitratio(file,sheets,label2ids,filelists,ratiolists):
    for sheet, label2id, filelist in zip(sheets, label2ids,filelists):
        data = pd.read_excel(file, header=None, sheet_name=sheet, skiprows=[0])
        data = shuffle(data)
        writetofile(filelist[0], data, label2id)
        for test_size in ratiolists:
            if test_size==0.4:
                train, test = train_test_split(data, random_state=0, test_size=0.4)
                dev, test = train_test_split(test, random_state=0, test_size=0.5)
                writetofile(filelist[1], train, label2id)
                writetofile(filelist[2], dev, label2id)
                writetofile(filelist[3], test, label2id)
            elif test_size == 0.3:
                train, test = train_test_split(data, random_state=0, test_size=0.3)
                dev, test = train_test_split(test, random_state=0, test_size=0.5)
                writetofile(filelist[4], train, label2id)
                writetofile(filelist[5], dev, label2id)
                writetofile(filelist[6], test, label2id)


if __name__ == '__main__':
    file = 'data/NLP聊天内容梳理2022.10.08.xlsx'
    splitratio(file, [0,1], label2idlist, filelists, [0.4,0.3])







