import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import gc
import tqdm
import cudf
from cuml.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import gc

class CFG:
    DATA_DIR = Path('../data/row')
    STAGE1_MODEL_NAME =  "sentence-transformers/all-MiniLM-L6-v2"
    STAGE1_PRETRAINED_PATH = Path('/root/workspace/Learning_Equality/dpr_all-MiniLM-L6-v2-finetuned/fold_0/epoch=3-step=1740.ckpt')
    STAGE2_MODEL_NAME = "xlm-roberta-base"
    STAGE2_MODEL_NAME = "xlm-roberta-base"
    STAGE2_PRETRAINED_PATH = Path('/root/workspace/Learning_Equality/xlm-roberta-base/fold_0/epoch_2-step_846.ckpt')
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    SEED=42
    N_NEIGHBORS = 100
    BATCH_PER_GPU = 96
    NUM_JOBS = 4
    THRESHOLD = 0.91
    TEST_PRED = False
  
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

class Stage1model(pl.LightningModule):
    def __init__(self, tokenizer,  model_name):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        config = AutoConfig.from_pretrained(model_name)

        self.topics_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.content_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.topics_encoder.resize_token_embeddings(len(self.tokenizer))
        self.content_encoder.resize_token_embeddings(len(self.tokenizer))
        
    def encode_topics(self, ids, mask):
        output_topics_embeddings = self.topics_encoder(ids, mask)
        return output_topics_embeddings
    
    def encode_content(self, ids, mask):
        output_content_enbeddings = self.content_encoder(ids, mask)
        return output_content_enbeddings
    
# データフレームをロードする
def read_data():
    topics_df = pd.read_csv(CFG.DATA_DIR / 'topics.csv')
    content_df = pd.read_csv(CFG.DATA_DIR /'content.csv')
    correlations_df = pd.read_csv(CFG.DATA_DIR /'correlations.csv')
    sample_submission = pd.read_csv(CFG.DATA_DIR / 'sample_submission.csv')
    content_df = content_df.fillna('')
    topics_df = topics_df.fillna('')
    topics_df['path'] = get_path_list(topics_df)
    topics_df = topics_df[topics_df.has_content].reset_index(drop=True)
    topics_df.drop(['channel', 'category', 'level', 'has_content'], axis = 1, inplace = True)
    content_df.drop(['kind',  'text', 'copyright_holder', 'license'], axis = 1, inplace = True)
    return topics_df, content_df, correlations_df, sample_submission

def get_train_test_data(topics_df, sample_submission):
    train_idx = topics_df[~topics_df.id.isin(sample_submission.topic_id)].index
    topics_df["fold"] = -1
    # 交差検証 用の番号を振ります。
    kf = KFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    for n, (_, val_index) in enumerate(kf.split(train_idx)):
        topics_df.loc[train_idx[val_index], "fold"] = int(n)
    topics_df["fold"] = topics_df["fold"]
    return topics_df

def get_topic_embeddings(model, tokenizer, data:pd.Series):
    data = list(data.fillna(''))
    gap = 1000
    
    token_outs = []
    # uniform dynamic padding
    for i in tqdm.tqdm(range(0, len(data), gap), desc='tokenization'):
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding=True, return_tensors='pt')
        token_outs.append(batch_tokens)
        
    outs = []
    model.to(CFG.DEVICE)
    model.eval()
    
    with torch.no_grad():
        for batch_tokens in tqdm.tqdm(token_outs, total=len(token_outs), desc='model output'):
            inputs = batch_tokens['input_ids'].to(CFG.DEVICE)
            attention_mask = batch_tokens['attention_mask'].to(CFG.DEVICE)
            out = model.encode_topics(inputs, attention_mask).last_hidden_state.mean(1)
            outs.append(out.cpu().numpy())
    return np.concatenate(outs)

def get_content_embeddings(model, tokenizer, data:pd.Series):
    data = list(data.fillna(''))
    gap = 1000
    
    token_outs = []
    # uniform dynamic padding
    for i in tqdm.tqdm(range(0, len(data), gap), desc='tokenization'):
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding=True, return_tensors='pt')
        token_outs.append(batch_tokens)
        
    outs = []
    model.to(CFG.DEVICE)
    model.eval()
    
    with torch.no_grad():
        for batch_tokens in tqdm.tqdm(token_outs, total=len(token_outs), desc='model output'):
            inputs = batch_tokens['input_ids'].to(CFG.DEVICE)
            attention_mask = batch_tokens['attention_mask'].to(CFG.DEVICE)
            out = model.encode_content(inputs, attention_mask).last_hidden_state.mean(1)
            outs.append(out.cpu().numpy())
    return np.concatenate(outs)

def get_df_from_knn_idx(knn_prd_idx, topics_df, content_df):
    all_content_ids =  content_df.id.to_numpy()
    all_topics_ids = topics_df.id.to_numpy()
    preds = []
    for t_idx in tqdm.tqdm(range(len(knn_prd_idx)), total=len(knn_prd_idx)):
        topic_id = all_topics_ids[t_idx]
        content_idx = knn_prd_idx.iloc[t_idx].to_numpy()
        content_idx = content_idx[~np.isnan(content_idx)].astype(int)
        content_ids = all_content_ids[content_idx]
        preds.append({
            'topic_id': topic_id,
            'content_ids': ' '.join(content_ids)
        })
    preds = pd.DataFrame.from_records(preds)
    return preds

def get_knn_predictions(model, tokenizer, topics_df, content_df):
    contents_embedding = get_content_embeddings(model, tokenizer, content_df.title)
    # contents_embedding = np.load('/root/workspace/Learning_Equality/SVM_DPR_paraphrase-multilingual-mpnet-base-v2-finetuned/train_data/contents_embedding.npy')
    topics_df['input_text'] = topics_df['title'] +"<|=t_sep=|>"+ topics_df['path']
    # topics_embedding = np.load('/root/workspace/Learning_Equality/SVM_DPR_paraphrase-multilingual-mpnet-base-v2-finetuned/train_data/topics_embedding.npy')
    topics_embedding = get_topic_embeddings(model, tokenizer, topics_df.input_text)
    contents_embedding_cudf = cudf.DataFrame(contents_embedding)
    topics_embedding_cudf = cudf.DataFrame(topics_embedding)
    all_topics_lang = topics_df.language.to_numpy()
    all_content_lang  =  content_df.language.to_numpy()
    lang_list = np.unique(all_topics_lang)
    knn_prd_idx = []
    for lang in lang_list:
        prd_data = topics_embedding_cudf[all_topics_lang==lang]
        fit_data = contents_embedding_cudf[all_content_lang==lang]
        print(f"言語:{lang}, コンテンツ数:{fit_data.shape[0]}, トピック数:{prd_data.shape[0]}")
        content_idx = np.where(all_content_lang==lang)[0]
        if fit_data.shape[0]>CFG.N_NEIGHBORS:
            n_neighbors= CFG.N_NEIGHBORS
        else:
            n_neighbors=int(fit_data.shape[0])
        model = NearestNeighbors(n_neighbors=n_neighbors)
        model.fit(fit_data) 
        _, prd_idx = model.kneighbors(prd_data)
        prd_idx.index=prd_data.index
        prd_idx = prd_idx.to_pandas().applymap(lambda x:content_idx[x])
        knn_prd_idx.append(prd_idx)
        del prd_data, fit_data, content_idx, prd_idx, model
        gc.collect()
    knn_prd_idx = pd.concat(knn_prd_idx, axis=0).sort_index()
    del contents_embedding, topics_embedding
    gc.collect()
    return knn_prd_idx

def build_trian_set(preds_df, topics_df, content_df, correlations_df, labels=False):
    topics_id2title = {id:title for id, title in zip(topics_df.id, topics_df.title)}
    content_id2title = {id:title for id, title in zip(content_df.id, content_df.title)}
    pred_contents = preds_df.content_ids.str.split(' ').tolist()
    pred_topic = preds_df.topic_id.to_numpy()
    all_folds = preds_df.fold.tolist()
    if labels:
        truth_contents = correlations_df.content_ids.str.split(' ').tolist()
    topic_title_list = []
    topic_id_list = []
    contents_title_list = []
    contents_id_list = []
    labels = []
    folds = []
    for idx, (c, t) in tqdm.tqdm(enumerate(zip(pred_contents, pred_topic)), total=len(pred_contents)):
        topic_id_list.extend([tt for tt in [t] * len(c)])
        topic_title_list.extend([topics_id2title[tt] for tt in [t] * len(c)])
        contents_id_list.extend(c)
        contents_title_list.extend([content_id2title[cc] for cc in c])
        if labels:
            label = np.isin(np.array(c), truth_contents[idx]) * 1
            labels.extend(label.tolist())
        folds.extend([all_folds[idx]] * len(c))
    if labels:
        return pd.DataFrame({'topic_id':topic_id_list, 'content_id':contents_id_list, 'topic_title':topic_title_list, 'content_title':contents_title_list, 'label':labels, 'fold':folds})
    else:
        return pd.DataFrame({'topic_id':topic_id_list, 'content_id':contents_id_list, 'topic_title':topic_title_list, 'content_title':contents_title_list, 'fold':folds})
    
def calc_AP_score(true_ids, pred_ids):
  true_positives = len(set(true_ids)&set(pred_ids))
  false_negatives = len(set(true_ids)-set(pred_ids))
  return true_positives/(true_positives + false_negatives)

def calc_AP_score_mean(target_df, pred_df):
  shape = target_df.shape
  score = [calc_AP_score(target_df.loc[i, 'content_ids'].split(), pred_df.loc[i, 'content_ids'].split()) for i in range(shape[0])]
  pred_df['score'] = score
  return pred_df['score'].mean()

def calc_f2_score(true_ids, pred_ids):
  true_positives = len(set(true_ids)&set(pred_ids))
  false_positives = len(set(pred_ids)-set(true_ids))
  false_negatives = len(set(true_ids)-set(pred_ids))

  beta = 2
  f2_score = ((1+beta**2)*true_positives)/((1+beta**2)*true_positives + beta**2*false_negatives + false_positives)
  return f2_score

def calc_f2_score_mean(target_df, pred_df):
  shape = target_df.shape
  score = [calc_f2_score(target_df.loc[i, 'content_ids'].split(), pred_df.loc[i, 'content_ids'].split()) for i in range(shape[0])]
  target_df['f2_score'] = score
  return target_df['f2_score'].mean()

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
    def __init__(self, model_name):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        
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

def inference_fn(test_loader, model):
    preds = []
    model.eval()
    model.to(CFG.DEVICE)
    tk0 = tqdm.tqdm(test_loader, total = len(test_loader))
    for inputs in tk0:
        input_ids = inputs['input_ids'].to(CFG.DEVICE)
        attention_mask = inputs['attention_mask'].to(CFG.DEVICE)
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask)[0]
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


def inference_stage1(topics_df, content_df, correlations_df, sample_submission):
    topics_df = get_train_test_data(topics_df,sample_submission)
    if CFG.TEST_PRED:
        idx = topics_df.id.isin(sample_submission.topic_id)
        topics_df = topics_df[idx].reset_index(drop=True)
        correlations_df = correlations_df[idx].reset_index(drop=True)
    else:
        idx = (topics_df.fold==0)
        topics_df = topics_df[idx].reset_index(drop=True)
        correlations_df = correlations_df[idx].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG.STAGE1_MODEL_NAME, use_fast=True)
    tokenizer.add_tokens(["<|=t_sep=|>"], special_tokens=True)
    model = Stage1model(tokenizer, CFG.STAGE1_MODEL_NAME)
    model.eval()
    model.load_state_dict(torch.load(CFG.STAGE1_PRETRAINED_PATH)["state_dict"])
    model.to(CFG.DEVICE)
    knn_prd_idx = get_knn_predictions(model, tokenizer, topics_df, content_df)
    preds_df = get_df_from_knn_idx(knn_prd_idx, topics_df, content_df)
    preds_df['fold'] = topics_df['fold']
    if not CFG.TEST_PRED:
        score = calc_AP_score_mean(correlations_df, preds_df)
        print(f'AP SCORE:{score}')
        df = build_trian_set(preds_df, topics_df, content_df, correlations_df, labels=False)
    else:
        df = build_trian_set(preds_df, topics_df, content_df, correlations_df, labels=False)
    del model, tokenizer
    gc.collect()
    return df, topics_df, content_df, correlations_df, sample_submission

def inference_stage2(df):
    tokenizer = AutoTokenizer.from_pretrained(CFG.STAGE2_MODEL_NAME)
    model = Stage2model(CFG.STAGE2_MODEL_NAME) 
    state = torch.load(CFG.STAGE2_PRETRAINED_PATH, map_location = torch.device('cpu'))
    model.load_state_dict(state['state_dict'])
    dataset = MyDataset(tokenizer, df, labels=False)
    func = lambda x:collate_fn(x, tokenizer)
    dataloader = DataLoader(dataset, batch_size=CFG.BATCH_PER_GPU, 
                            collate_fn=func,
                            shuffle=False, num_workers=CFG.NUM_JOBS, drop_last=False)
    preds = inference_fn(dataloader,model)
    return preds

def postprocess(df, threshold: float=0.9, top_n: int = 5):
    df.loc[df.predictions_proba>=threshold, 'pred'] = 1
    df.loc[df.predictions_proba<threshold, 'pred'] = 0 
    result = []
    grouped_df = df.groupby('topic_id')
    for idx, df_ in tqdm.tqdm(grouped_df, total=len(grouped_df)):
        df_ = df_.sort_values('predictions_proba', ascending=False)
        if df_.pred.sum()==0:
            res_df = df_.iloc[:5]
        else:
            res_df = df_[df_.pred==1]
        result.append(res_df.loc[:,['topic_id', 'content_id']])
    result = pd.concat(result, axis=0)
    result = pd.DataFrame(result.groupby('topic_id').apply(lambda x:' '.join(x.content_id)))
    result =result.reset_index().rename(columns={0:'content_ids'})
    result['content_ids'] = result['content_ids'].apply(lambda x:' '.join(x.split(' ')))
    return result

def main():
    topics_df, content_df, correlations_df, sample_submission = read_data()
    df, topics_df, content_df, correlations_df, sample_submission = inference_stage1(topics_df, content_df, correlations_df, sample_submission)    
    preds = inference_stage2(df)
    df['predictions_proba'] = preds
    
    if not CFG.TEST_PRED:
        df.to_csv("oof_df.csv", index=False)
        target_df = correlations_df[correlations_df.topic_id.isin(df.topic_id)].reset_index(drop=True)
        result = postprocess(df, threshold=CFG.THRESHOLD, top_n = 5)
        score = calc_f2_score_mean(target_df, result)
        print(f"threshold:{CFG.THRESHOLD} score:{score}")
    else:
        submission_df = postprocess(df, threshold=CFG.THRESHOLD, top_n = 5)
        submission_df.to_csv('submission.csv', index=False)
if __name__ == '__main__':
    main()
    