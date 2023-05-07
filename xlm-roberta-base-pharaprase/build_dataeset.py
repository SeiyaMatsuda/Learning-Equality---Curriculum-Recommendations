import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import gc
import tqdm
import cudf
from cuml.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from pathlib import Path
import pytorch_lightning as pl

class CFG:
  MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  PRETRAINED_PATH = Path('./epoch_1_step_434.ckpt')
  DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
  SEED=42
  TRAIN_STEP = True
  N_NEIGHBORS = 100
  DATSET_BUILD = True
  
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
    topics_df = pd.read_csv('../data/row/topics.csv')
    content_df = pd.read_csv('../data/row/content.csv')
    correlations_df = pd.read_csv('../data/row/correlations.csv')
    sample_submission = pd.read_csv('../data/row/sample_submission.csv')
    content_df = content_df.fillna('')
    topics_df = topics_df.fillna('')
    topics_df['path'] = get_path_list(topics_df)
    topics_df = topics_df[topics_df.has_content].reset_index(drop=True)
    topics_df.drop(['channel', 'category', 'level', 'has_content'], axis = 1, inplace = True)
    content_df.drop(['kind', 'copyright_holder', 'license'], axis = 1, inplace = True)
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
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding="max_length", return_tensors='pt', max_length=128)
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
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding="max_length", return_tensors='pt', max_length=128)
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
    content_df['input_text']  = content_df['title'] +"<|=t_sep=|>"+ content_df['description']+"<|=t_sep=|>"+ content_df['text']
    contents_embedding = get_content_embeddings(model, tokenizer, content_df.input_text)
    topics_df['input_text'] = topics_df['path']+ "<|=t_sep=|>" + topics_df['title'] +"<|=t_sep=|>"+ topics_df['description']
    topics_embedding = get_topic_embeddings(model, tokenizer, topics_df.input_text)
    contents_embedding_cudf = cudf.DataFrame(contents_embedding)
    topics_embedding_cudf = cudf.DataFrame(topics_embedding)
    del contents_embedding,topics_embedding
    gc.collect()
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
    gc.collect()
    return knn_prd_idx

def build_trian_set(preds_df, topics_df, content_df, correlations_df):
    topics_id2title = {id:title for id, title in zip(topics_df.id, topics_df.title)}
    content_id2title = {id:title for id, title in zip(content_df.id, content_df.title)}
    pred_contents = preds_df.content_ids.str.split(' ').tolist()
    pred_topic = preds_df.topic_id.to_numpy()
    all_folds = preds_df.fold.tolist()
    truth_contents = correlations_df.content_ids.str.split(' ').tolist()
    topic_title_list = []
    contents_title_list = []
    labels = []
    folds = []
    for idx, (c, t) in tqdm.tqdm(enumerate(zip(pred_contents, pred_topic)), total=len(pred_contents)):
        topic_title_list.extend([topics_id2title[tt] for tt in [t] * len(c)])
        contents_title_list.extend([content_id2title[cc] for cc in c])
        label = np.isin(np.array(c), truth_contents[idx]) * 1
        labels.extend(label.tolist())
        folds.extend([all_folds[idx]] * len(c))
    return pd.DataFrame({'topic_title':topic_title_list, 'content_title':contents_title_list, 'label':labels, 'fold':folds})
    
    
def calc_score(true_ids, pred_ids):
  true_positives = len(set(true_ids)&set(pred_ids))
  false_negatives = len(set(true_ids)-set(pred_ids))
  return true_positives/(true_positives + false_negatives)

def calc_score_mean(target_df, pred_df):
  shape = target_df.shape
  score = [calc_score(target_df.loc[i, 'content_ids'].split(), pred_df.loc[i, 'content_ids'].split()) for i in range(shape[0])]
  pred_df['score'] = score
  return pred_df['score'].mean()

def main():
    topics_df, content_df, correlations_df, sample_submission = read_data()
    topics_df = get_train_test_data(topics_df,sample_submission)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)
    tokenizer.add_tokens(["<|=t_sep=|>"], special_tokens=True)
    model = Stage1model(tokenizer, CFG.MODEL_NAME)
    model.load_state_dict(torch.load(CFG.PRETRAINED_PATH)["state_dict"])
    knn_prd_idx = get_knn_predictions(model, tokenizer, topics_df, content_df)
    preds_df = get_df_from_knn_idx(knn_prd_idx, topics_df, content_df)
    preds_df['fold'] = topics_df['fold']
    score = calc_score_mean(correlations_df, preds_df)
    print(f'AP SCORE:{score}')
    if CFG.DATSET_BUILD:
        train_df = build_trian_set(preds_df, topics_df, content_df, correlations_df)
        train_df.to_csv('train_df.csv')

if __name__=='__main__':
    main()
