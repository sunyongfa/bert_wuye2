import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model_definition.my_fastbert import FastBertModel, BertConfig
from data_utils.dataset_preparing import PrepareDataset, TextCollate
import torch.nn.functional as F
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model

if __name__ == "__main__":

    eval_dataset = PrepareDataset(vocab_file="./model_pretrained/bert-chinese/bert-pytorch-google/vocab.txt",
                                 max_seq_len=128,
                                 num_class=4,
                                 data_file="./data/sentiment/test1/dev.tsv")
    dataloader = data.DataLoader(dataset=eval_dataset,
                                 collate_fn=TextCollate(eval_dataset),
                                 pin_memory=False,
                                 batch_size=1,
                                 num_workers=2,
                                 shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, unit="batch", ncols=100, desc="Evaluating process: ")):
        texts = batch["texts"]
        tokens = batch["tokens"]
        segment_ids = batch["segment_ids"]
        attn_masks = batch["attn_masks"]
        labels = batch["labels"]
        if(False and step>50):
            break
        print(step)
        print(texts)
