import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import random
print(torch.cuda.is_available)
class CFG:
    ROW_DIR = Path('../data/row')
    PROCESSED_DIR = Path('../data/processed/train_data')
    TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_PER_GPU = 768
    MAX_LENGTH = 128
    SEED=42
    NUM_EPOCHS=12
    LR = 5e-5
    NUM_GPUS=1
    NUM_JOBS=4
    N_FOLD = 1
    AMP=True

def get_train_test_data(train, train_idx):
    train["fold"] = -1
    # 交差検証 用の番号を振ります。
    kf = KFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    for n, (train_index, val_index) in enumerate(kf.split(train_idx)):
        train.loc[train_idx[val_index], "fold"] = int(n)
    train["fold"] = train["fold"]
    return train

def get_path_list(df):
    topics_id2title = {k:v for k, v in zip(df.id.to_list(), df.title.to_list())}
    topics_id2description = {k:v for k, v in zip(df.id.to_list(), df.description.to_list())}
    topics_id2parents = {k:v for k, v in zip(df.id.to_list(), df.parent.to_list())}
    path_list = []
    for id in df.id.to_list():
        res_list = []
        while True:
            res_list.append(topics_id2title[id])
            id = topics_id2parents[id]
            if id=="":
                break
        path_list.append(" | ".join(res_list[::-1][:-1]))
    return path_list
    
class MyDataset(Dataset):
    def __init__(self, tokenizer, df, topics_df, contetn_df):
        self.all_topics_id = df.topic_id.to_numpy()
        self.all_content_id = df.content_id.to_numpy()
        self.topics_title_dict = {id:title for id, title in zip(topics_df.id, topics_df.title)}
        self.topics_description_dict = {id:description for id, description in zip(topics_df.id, topics_df.description)}
        self.topics_path_dict = {id:path for id, path in zip(topics_df.id, topics_df.path)}
        self.content_title_dict = {id:title for id, title in zip(contetn_df.id, contetn_df.title)}
        self.content_description_dict = {id:description for id, description in zip(contetn_df.id, contetn_df.description)}
        self.content_text_dict = {id:text for id, text in zip(contetn_df.id, contetn_df.text)}
        self.all_topics_title = [self.topics_title_dict[id] for id in self.all_topics_id]
        self.all_topics_description = [self.topics_description_dict[id] for id in self.all_topics_id]
        self.all_topics_path = [self.topics_path_dict[id] for id in self.all_topics_id]
        self.all_content_title = [self.content_title_dict[id] for id in self.all_content_id]
        self.all_content_description = [self.content_description_dict[id] for id in self.all_content_id]
        self.all_content_text = [self.content_text_dict[id] for id in self.all_content_id]
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.all_topics_id)
    def __getitem__(self, idx):
        topic_id = self.all_topics_id[idx]
        content_id = self.all_content_id[idx]
        topic_title = self.all_topics_title[idx]
        topic_description = self.all_topics_description[idx]
        topic_path = self.all_topics_path[idx]
        topic_rep = topic_path + "<|=t_sep=|>" + topic_title + "<|=t_sep=|>" + topic_description 
        content_title = self.all_content_title[idx]
        content_description = self.all_content_description[idx]
        content_text = self.all_content_text[idx]
        content_rep = content_title + "<|=t_sep=|>" + content_description + "<|=t_sep=|>" + content_text
        topic_inputs =  self.tokenizer.encode_plus(
        topic_rep, 
        return_tensors = None, 
        add_special_tokens = True,
        )
        content_inputs =  self.tokenizer.encode_plus(
        content_rep, 
        return_tensors = None, 
        add_special_tokens = True,
        )
        topic_input_ids = topic_inputs['input_ids']
        topic_attention_mask = topic_inputs['attention_mask']
        content_input_ids = content_inputs['input_ids']
        content_attention_mask = content_inputs['attention_mask']
        return {
                'topic_input_ids':topic_input_ids,
                'topic_attention_mask':topic_attention_mask,
                'content_input_ids':content_input_ids,
                'content_attention_mask':content_attention_mask,
                }

def collate_fn(batch, tokenizer, max_length=512):
    """
    自然言語処理タスク向けのcollate_fn
    """
    topic_max_len = max([len(b['topic_input_ids']) for b in batch])
    if topic_max_len>max_length:
        topic_max_len = max_length
    content_max_len = max([len(b['content_input_ids']) for b in batch])
    if content_max_len>max_length:
        content_max_len = max_length
    # バッチ内の各要素から文章とラベルを取得
    topic_input_ids = [b['topic_input_ids']+[tokenizer.pad_token_id] * (topic_max_len-len(b['topic_input_ids'])) 
                       if len(b['topic_input_ids'])<topic_max_len else b['topic_input_ids'][:topic_max_len] for b in batch]
    topic_attention_mask = [b['topic_attention_mask']+[0] * (topic_max_len-len(b['topic_attention_mask'])) 
                            if len(b['topic_attention_mask'])<topic_max_len else b['topic_attention_mask'][:topic_max_len] for b in batch]
    content_input_ids = [b['content_input_ids']+[tokenizer.pad_token_id] * (content_max_len-len(b['content_input_ids'])) 
                         if len(b['content_input_ids'])<content_max_len else b['content_input_ids'][:content_max_len] for b in batch] 
    content_attention_mask = [b['content_attention_mask']+[0] * (content_max_len-len(b['content_attention_mask'])) 
                              if len(b['content_attention_mask'])<content_max_len else b['content_attention_mask'][:content_max_len] for b in batch] 
    return {
              'topic_input_ids':torch.tensor(topic_input_ids, dtype=torch.long),
              'topic_attention_mask':torch.tensor(topic_attention_mask, dtype=torch.long),
              'content_input_ids':torch.tensor(content_input_ids, dtype=torch.long),
              'content_attention_mask':torch.tensor(content_attention_mask, dtype=torch.long),
              }
    
class FeedbackModel(pl.LightningModule):
    def __init__(self, tokenizer,  model_name, learning_rate, num_train_steps, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"
        
        config = AutoConfig.from_pretrained(model_name)

        self.topics_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.content_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.topics_encoder.resize_token_embeddings(len(self.tokenizer))
        self.content_encoder.resize_token_embeddings(len(self.tokenizer))
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(self, ids1, mask1, ids2, mask2):
        output_topics_encoder = self.topics_encoder(ids1, mask1)
        output_content_encoder = self.content_encoder(ids2, mask2)
        topics_embeddings = output_topics_encoder.last_hidden_state.mean(1)
        content_embeddings = output_content_encoder.last_hidden_state.mean(1)
        loss = self.loss(topics_embeddings, content_embeddings)
        return topics_embeddings, content_embeddings, loss
    
    def loss(self, embeddings_a, embeddings_b):
        loss = self.MultipleNegativeRankingLoss(embeddings_a, embeddings_b)
        return loss
    

    def cos_sim(self, a, b):
        # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))
    
    # def MultipleNegativeRankingLoss(self, embeddings_a:torch.Tensor, embeddings_b:torch.Tensor):
    #     """
    #     Computes MultipleNegativeRankingLoss
    #     """

    #     scores = torch.matmul(embeddings_a, embeddings_b.t())
    #     diagonal_mean = torch.mean(torch.diag(scores))
    #     mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
    #     return -diagonal_mean + mean_log_row_sum_exp
    
    def MultipleNegativeRankingLoss(self, embeddings_a:torch.Tensor, embeddings_b:torch.Tensor):
        similarity_scores = (
            self.cos_sim(embeddings_a, embeddings_b) * 20.0
        )  # Not too sure why to scale it by 20: https://github.com/UKPLab/sentence-transformers/blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L57
        
        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)
    
    def training_step(self, batch, batch_idx):
        ids1, mask1, ids2, mask2 = batch['topic_input_ids'], batch['topic_attention_mask'], batch['content_input_ids'], batch['content_attention_mask']
        _, _, loss = self.forward(ids1=ids1, mask1=mask1,ids2=ids2, mask2=mask2)
        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids1, mask1, ids2, mask2 = batch['topic_input_ids'], batch['topic_attention_mask'], batch['content_input_ids'], batch['content_attention_mask']
        _, _, loss = self.forward(ids1=ids1, mask1=mask1,ids2=ids2, mask2=mask2)
        self.log('val_loss', loss, on_step=True, logger=True, prog_bar=True)
        return {'val_loss': loss}
        
    def validation_epoch_end(self, val_step_outputs):
        val_loss = sum([val['val_loss'] for val in val_step_outputs])/len([val['val_loss'] for val in val_step_outputs])
        self.log("val_loss", val_loss, on_epoch=True, logger=True, prog_bar=False)
        return {'multi_negative_ranking_loss':val_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * self.num_train_steps), num_training_steps=self.num_train_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

def train_fold(df, tokenizer, topics_df, content_df, fold):
    pl.seed_everything(CFG.SEED)

    dirpath = f'./fold_{fold}'
    os.makedirs(dirpath, exist_ok=True)

    print(f'================================== Prepare Data for fold{fold} =====================================')
    train_samples = df[df["fold"] != fold].reset_index(drop=True)
    valid_samples = df[df["fold"] == fold].reset_index(drop=True)
    train_dataset = MyDataset(tokenizer, train_samples, topics_df, content_df)
    valid_dataset = MyDataset(tokenizer, valid_samples, topics_df, content_df)
    func = lambda x:collate_fn(x, tokenizer, CFG.MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.BATCH_PER_GPU, 
                                  collate_fn=func,
                                  shuffle=True, num_workers=CFG.NUM_JOBS, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=CFG.BATCH_PER_GPU, 
                                collate_fn=func,
                                shuffle=False, num_workers=CFG.NUM_JOBS, drop_last=False)

    total_batch_size = CFG.BATCH_PER_GPU * CFG.NUM_GPUS
    steps_per_epoch = int(len(train_dataset) // total_batch_size)
    num_train_step = int(steps_per_epoch * CFG.NUM_EPOCHS)
    lightning_model = FeedbackModel(tokenizer, CFG.MODEL, CFG.LR, num_train_step, steps_per_epoch)

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
        dirpath=dirpath,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=2,
        mode="min",
    )
    logger_tb = pl.loggers.TensorBoardLogger("logs", name="stage2model_log")
    
    call_backs = [checkpoint, lr_monitor, early_stopping]

    trainer = pl.Trainer(
        max_epochs=CFG.NUM_EPOCHS,
        logger = [logger_tb],
        callbacks=call_backs,
        gpus=-1 if CFG.NUM_GPUS != 1 else [0],
        strategy="ddp" if CFG.NUM_GPUS != 1 else None,
        precision = 16 if CFG.AMP else 32,
        amp_backend = "native",
    )
    print(f'================================== Start Training fold{fold} =====================================')
    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    best_model_path = checkpoint.best_model_path
    print("best model path: ", best_model_path)
    
def main():
    print(f"================================== Start Running =====================================")
    topics_df = pd.read_csv(CFG.ROW_DIR / 'topics.csv')
    content_df = pd.read_csv(CFG.ROW_DIR / 'content.csv')
    sample_submission = pd.read_csv(CFG.ROW_DIR / 'sample_submission.csv')
    correlations_df = pd.read_csv(CFG.ROW_DIR / 'correlations.csv')
    content_df = content_df.fillna('')
    topics_df = topics_df.fillna('')
    # pathを取得する
    topics_df['path'] = get_path_list(topics_df)
    topics_df = topics_df[topics_df.has_content].reset_index(drop=True)
    topics_df.drop(['channel', 'category', 'level', 'parent', 'has_content'], axis = 1, inplace = True)
    content_df.drop(['kind',  'copyright_holder', 'license'], axis = 1, inplace = True)
    train_idx = topics_df[~topics_df.id.isin(sample_submission.topic_id)].index
    topics_df = get_train_test_data(topics_df, train_idx)
    topics_df.to_csv('./train_topics.csv')
    train_df = topics_df.join(correlations_df.content_ids.str.split(" ").explode())
    train_df= train_df.reset_index(drop=True)
    train_df = train_df.drop_duplicates(subset=['id', 'content_ids'])
    train_df = train_df.rename(columns={'id':'topic_id', 'content_ids':'content_id'}).drop(columns = ['title', 'language'])
    train_df = train_df[train_df.fold!=-1]
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)
    tokenizer.add_tokens(["<|=t_sep=|>"], special_tokens=True)

    for fold in range(CFG.N_FOLD):
        train_fold(train_df, tokenizer, topics_df, content_df, fold)

if __name__ == "__main__":
    main()
