import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import random
import bitsandbytes as bnb
print(torch.cuda.is_available)
class CFG:
    ROW_DIR = Path('../data/row')
    TRAIN_DATA_PATH = 'train_df.csv'
    TOKENIZER = "xlm-roberta-base"
    MODEL = "xlm-roberta-base"
    BATCH_PER_GPU = 256
    SEED=42
    NUM_EPOCHS=12
    LR = 1e-5
    NUM_GPUS=1
    NUM_JOBS=4
    ACUUMULATE_BATCH=4
    AMP=True
    N_FOLD = 1
    
class MyDataset(Dataset):
    def __init__(self, tokenizer, df, labels:bool=True):
        self.all_topics_title = df.topic_title.to_numpy()
        self.all_content_title = df.content_title.to_numpy()
        self.tokenizer = tokenizer
        self.labels=labels
        if self.labels:
          self.all_labels = df.label.to_numpy()
    def __len__(self):
        return len(self.all_topics_title)
    def __getitem__(self, idx):
        topic_text = self.all_topics_title[idx]
        content_text = self.all_content_title[idx]
        inputs =  self.tokenizer.encode_plus(
        topic_text + '[SEP]' + content_text, 
        return_tensors = None, 
        add_special_tokens = True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if self.labels:
            label = self.all_labels[idx]
        else:
            label=-1
        return {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'labels':label
                }

def collate_fn(batch, tokenizer):
    """
    自然言語処理タスク向けのcollate_fn
    """
    max_len = max([len(b['input_ids']) for b in batch])
    # バッチ内の各要素から文章とラベルを取得
    input_ids = [b['input_ids']+[tokenizer.pad_token_id] * (max_len-len(b['input_ids'])) for b in batch]
    attention_mask = [b['attention_mask']+[0] * (max_len-len(b['attention_mask'])) for b in batch]
    labels = [b['labels'] for b in batch]
    return {
              'input_ids':torch.tensor(input_ids, dtype=torch.long),
              'attention_mask':torch.tensor(attention_mask, dtype=torch.long),
              'labels':torch.tensor(labels, dtype=torch.long),
              }
    
class Stage2model(pl.LightningModule):
    def __init__(self, model_name, learning_rate, num_train_steps, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"
        
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        
        config = AutoConfig.from_pretrained(model_name, num_labels=1)
        
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, 1)
        
        self.loss_fn = nn.BCEWithLogitsLoss(reduction = "mean")
    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        last_hidden_states = transformer_out.last_hidden_state
        sequence_output = self.dropout(last_hidden_states)
        sequence_output = sequence_output[:,0,:]
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = logits.squeeze()
        loss = 0
        if targets is not None:
            loss1 = self.loss(logits1.squeeze(), targets)
            loss2 = self.loss(logits2.squeeze(), targets)
            loss3 = self.loss(logits3.squeeze(), targets)
            loss4 = self.loss(logits4.squeeze(), targets)
            loss5 = self.loss(logits5.squeeze(), targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            return logits, loss

        return logits, loss
    
    def loss(self,  outputs, targets):
        loss_fn = nn.BCEWithLogitsLoss(reduction = "mean")
        loss = self.loss_fn(outputs, targets.float())
        return loss
    def monitor_metrics(self, outputs, targets):
        active_logits = outputs.view(-1, 1)
        true_labels = targets.view(-1).cpu().numpy()
        f2_score = self.f2_score(active_logits.data.cpu(), true_labels, threshold=0.5, beta=2)
        return f2_score    
    
    def f2_score(self, logit, label, threshold=0.5, beta=2):
        prob = torch.sigmoid(logit)
        prob = prob > threshold
        label = label > threshold

        TP = (prob & label).sum().float()
        TN = ((~prob) & (~label)).sum().float()
        FP = (prob & (~label)).sum().float()
        FN = ((~prob) & label).sum().float()

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        return F2.mean(0)
    
    def training_step(self, batch, batch_idx):
        ids, mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
        _, loss = self.forward(ids=ids, mask=mask, targets=targets)
        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids, mask, targets= batch['input_ids'], batch['attention_mask'], batch['labels']
        logits, loss = self.forward(ids=ids, mask=mask, targets=targets)
        self.log('val_loss', loss, on_step=True, logger=True, prog_bar=True)
        return {'val_loss': loss, 'logits':logits, 'targets':targets}
        
    def validation_epoch_end(self, val_step_outputs):
        all_logits = torch.concat([val['logits'] for val in val_step_outputs], dim=0)
        all_targets = torch.concat([val['targets'] for val in val_step_outputs], dim=0)
        threholds = np.arange(0.05, 0.95, 0.05)
        f2_score = np.array([self.f2_score(all_logits, all_targets, threshold=thre, beta=2).data.cpu() for thre in threholds])
        self.log("f2_score", {"max_score": np.max(f2_score), "threshold": threholds[np.argmax(f2_score)]}, on_epoch=True, logger=True, prog_bar=False)
        return {'f2_score': max(f2_score)}
    
    def configure_optimizers(self):
        # optimizer =  bnb.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * self.num_train_steps), num_training_steps=self.num_train_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
def under_sampling(df: pd.DataFrame):

    #order_of_finish = 着順
    low_frequency_data_sample = df[df["label"] == 1]
    low_frequency_data_size = len(low_frequency_data_sample)
    # 高頻度データの行ラベル
    high_frequency_data = df[df["label"] == 0].index

    # 高頻度データの行ラベルから、低頻度のデータと同じ数をランダムで抽出
    random_indices = np.random.choice(high_frequency_data, low_frequency_data_size, replace=False)

    # 抽出した行ラベルを使って、該当するデータを取得
    high_frequency_data_sample = df.loc[random_indices]
    # データをマージする。 concatは結合API
    merged_data = pd.concat([high_frequency_data_sample, low_frequency_data_sample], ignore_index=True)
    balanced_data = pd.DataFrame(merged_data)
    return balanced_data

def train_fold(df, tokenizer, fold):
    pl.seed_everything(CFG.SEED)

    dirpath = f'./fold_{fold}'
    os.makedirs(dirpath, exist_ok=True)

    print(f'================================== Prepare Data for fold{fold} =====================================')
    train_samples = df[df["fold"] != fold].reset_index(drop=True)
    valid_samples = df[df["fold"] == fold].reset_index(drop=True)
    train_dataset = MyDataset(tokenizer, under_sampling(train_samples))
    valid_dataset = MyDataset(tokenizer, under_sampling(valid_samples))
    func = lambda x:collate_fn(x, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.BATCH_PER_GPU, 
                                  collate_fn=func,
                                  shuffle=True, num_workers=CFG.NUM_JOBS, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=CFG.BATCH_PER_GPU, 
                                collate_fn=func,
                                shuffle=False, num_workers=CFG.NUM_JOBS, drop_last=False)

    total_batch_size = CFG.BATCH_PER_GPU * CFG.NUM_GPUS
    steps_per_epoch = int(len(train_dataset) // total_batch_size)
    num_train_step = int(steps_per_epoch * CFG.NUM_EPOCHS)
    lightning_model = Stage2model(CFG.MODEL, CFG.LR, num_train_step, steps_per_epoch)

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
        patience=4,
        mode="min",
    )
    logger_tb = pl.loggers.TensorBoardLogger("logs", name="stage2model_log")
    
    call_backs = [checkpoint, lr_monitor, early_stopping]

    trainer = pl.Trainer(
        max_epochs=CFG.NUM_EPOCHS,
        logger=[logger_tb],
        callbacks=call_backs,
        gpus=-1 if CFG.NUM_GPUS != 1 else [0],
        strategy="ddp" if CFG.NUM_GPUS != 1 else None,
        precision = 16 if CFG.AMP else 32,
        amp_backend = "native",
        accumulate_grad_batches=CFG.ACUUMULATE_BATCH,

    )
    print(f'================================== Start Training fold{fold} =====================================')
    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    best_model_path = checkpoint.best_model_path
    print("best model path: ", best_model_path)
    
def main():
    print(f"================================== Start Running =====================================")
    train_df = pd.read_csv(CFG.TRAIN_DATA_PATH, index_col=0)
    train_df = train_df[~train_df.topic_title.isna()]
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)
    for fold in range(CFG.N_FOLD):
        train_fold(train_df, tokenizer, fold)

if __name__ == "__main__":
    main()
