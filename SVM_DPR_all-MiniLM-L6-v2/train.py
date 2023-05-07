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
import pickle
import cupy as cp
from cuml.svm import SVC
import cuml
from cuml.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

class CFG:
    DATA_DIR = Path('../data/row')
    STAGE1_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    STAGE1_PRETRAINED_PATH = Path('./epoch_8_step_2610.ckpt')
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    SEED=42
    N_NEIGHBORS = 100
    N_FOLDS=5
    BATCH_SIZE = 25000
    NUM_JOBS = 4
    THRESHOLD = 0.91
    SAVE_DIR = Path('./svm_model')
    MAX_LEN = 128
    
def init_logger(log_file='train.log'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

Path("logs").mkdir(parents=True, exist_ok=True)
LOGGER = init_logger("logs/train.log")

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
    outs = []
    model.to(CFG.DEVICE)
    model.eval()
    # uniform dynamic padding
    for i in tqdm.tqdm(range(0, len(data), gap), desc='tokenization'):
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding="max_length", return_tensors='pt', max_length=CFG.MAX_LEN)
        with torch.no_grad():
            inputs = batch_tokens['input_ids'].to(CFG.DEVICE)
            attention_mask = batch_tokens['attention_mask'].to(CFG.DEVICE)
            out = model.encode_topics(inputs, attention_mask).last_hidden_state.mean(1)
            outs.append(out.cpu().numpy())
            del batch_tokens, inputs, attention_mask
            gc.collect()
    return np.concatenate(outs)

def get_content_embeddings(model, tokenizer, data:pd.Series):
    data = list(data.fillna(''))
    gap = 1000
    outs = []
    # uniform dynamic padding
    model.to(CFG.DEVICE)
    model.eval()
    for i in tqdm.tqdm(range(0, len(data), gap), desc='tokenization'):
        batch_tokens=tokenizer(data[i:i+gap], truncation=True, padding="max_length", return_tensors='pt', max_length=CFG.MAX_LEN)
        with torch.no_grad():
            inputs = batch_tokens['input_ids'].to(CFG.DEVICE)
            attention_mask = batch_tokens['attention_mask'].to(CFG.DEVICE)
            out = model.encode_content(inputs, attention_mask).last_hidden_state.mean(1)
            outs.append(out.cpu().numpy())
            del batch_tokens, inputs, attention_mask
            gc.collect()
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
    # np.save('contents_embedding', contents_embedding)
    # contents_embedding = np.load('contents_embedding.npy')
    topics_df['input_text'] = topics_df['path']+ "<|=t_sep=|>" + topics_df['title'] +"<|=t_sep=|>"+ topics_df['description']
    topics_embedding = get_topic_embeddings(model, tokenizer, topics_df.input_text)
    # np.save('topics_embedding', topics_embedding)
    # topics_embedding = np.load('topics_embedding.npy')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
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
        LOGGER.info(f"言語:{lang}, コンテンツ数:{fit_data.shape[0]}, トピック数:{prd_data.shape[0]}")
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
    return knn_prd_idx, topics_embedding_cudf, contents_embedding_cudf

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
    ground_truth = []
    folds = []
    for idx, (c, t) in tqdm.tqdm(enumerate(zip(pred_contents, pred_topic)), total=len(pred_contents)):
        topic_id_list.extend([tt for tt in [t] * len(c)])
        topic_title_list.extend([topics_id2title[tt] for tt in [t] * len(c)])
        contents_id_list.extend(c)
        contents_title_list.extend([content_id2title[cc] for cc in c])
        if labels:
            gt_label = np.isin(np.array(c), truth_contents[idx]) * 1
            ground_truth.extend(gt_label.tolist())
        folds.extend([all_folds[idx]] * len(c))
    if labels:
        return pd.DataFrame({'topic_id':topic_id_list, 'content_id':contents_id_list, 'topic_title':topic_title_list, 'content_title':contents_title_list, 'label':ground_truth, 'fold':folds})
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


def inference_stage1(topics_df, content_df, correlations_df, sample_submission):
    topics_df = get_train_test_data(topics_df,sample_submission)
    idx = (topics_df.fold!=-1)
    topics_df = topics_df[idx].reset_index(drop=True)
    correlations_df = correlations_df[idx].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG.STAGE1_MODEL_NAME, use_fast=True)
    tokenizer.add_tokens(["<|=t_sep=|>"], special_tokens=True)
    model = Stage1model(tokenizer, CFG.STAGE1_MODEL_NAME)
    model.load_state_dict(torch.load(CFG.STAGE1_PRETRAINED_PATH)["state_dict"])
    model.eval()
    model.to(CFG.DEVICE)
    knn_prd_idx, topics_embedding_cudf, contents_embedding_cudf = get_knn_predictions(model, tokenizer, topics_df, content_df)
    preds_df = get_df_from_knn_idx(knn_prd_idx, topics_df, content_df)
    preds_df['fold'] = topics_df['fold']
    df = build_trian_set(preds_df, topics_df, content_df, correlations_df, labels=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return df, topics_df, content_df, correlations_df, sample_submission, topics_embedding_cudf, contents_embedding_cudf 

def train_fn(df, fold, to_idx, co_idx, topics_embedding_cudf,contents_embedding_cudf, save_dir):
    tr_idx = df.index.to_numpy()
    tr_y = (df.label * 1).to_numpy()
    rus = RandomUnderSampler(random_state=42)
    tr_idx, tr_y = rus.fit_resample(tr_idx.reshape(-1, 1), tr_y)
    tr_idx = cp.array(tr_idx)
    tr_idx, tr_y = cp.array(tr_idx), cp.array(tr_y)
    tr_idx = tr_idx.reshape(-1)
    tr_X_topic =  topics_embedding_cudf.iloc[to_idx[tr_idx]].values
    tr_X_content =  contents_embedding_cudf.iloc[co_idx[tr_idx]].values
    tr_X = cp.hstack((tr_X_topic, tr_X_content))
    del tr_X_topic, tr_X_content
    gc.collect()
    LOGGER.info(f'====training fold:{fold}====')
    scaler = MinMaxScaler()
    pca = cuml.PCA(n_components=500)
    tr_X = pca.fit_transform(tr_X)
    tr_X = scaler.fit_transform(tr_X)
    model = SVC(kernel='rbf', C=10, gamma=1, verbose=True, probability=True, max_iter=1000)
    model.fit(tr_X, tr_y)
    del tr_X
    gc.collect()
    with open(save_dir, 'wb') as f:
        pickle.dump({'pca':pca,
                    'scaler':scaler, 
                    'model':model}, f)
    
def inference_stage2(df, to_idx, co_idx, topics_embedding_cudf, contents_embedding_cudf, save_model_path):
    pipline = pickle.load(open(save_model_path, mode='rb'))
    pca = pipline['pca']
    scaler = pipline['scaler']
    model = pipline['model']
    test_idx = df.index.to_numpy()
    preds = []
    for test_idx_split in tqdm.tqdm(np.array_split(test_idx, int(len(test_idx)/CFG.BATCH_SIZE)+1), total=int(len(test_idx)/CFG.BATCH_SIZE)+1):
        test_X_topic =  topics_embedding_cudf.iloc[to_idx[test_idx_split]].values
        test_X_content =  contents_embedding_cudf.iloc[co_idx[test_idx_split]].values
        test_X = cp.hstack((test_X_topic, test_X_content))
        del test_X_topic, test_X_content
        gc.collect()
        test_X = pca.transform(test_X)
        test_X = scaler.transform(test_X)
        test_pred = model.predict_proba(test_X)
        preds.append(test_pred.get()[:,1])
    predictions = np.concatenate(preds)
    return predictions

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
    df, topics_df, content_df, correlations_df, sample_submission, topics_embedding_cudf, contents_embedding_cudf = inference_stage1(topics_df, content_df, correlations_df, sample_submission)  
    scores = []
    CFG.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    topics_index2id = topics_df.id.to_dict()
    topics_id2index = dict(zip(topics_index2id.values(), topics_index2id.keys()))
    contents_index2id = content_df.id.to_dict()
    contents_id2index = dict(zip(contents_index2id.values(), contents_index2id.keys()))
    to_idx = cp.array(df.topic_id.apply(lambda x:topics_id2index[x]).to_numpy())
    co_idx = cp.array(df.content_id.apply(lambda x:contents_id2index[x]).to_numpy())
    for fold in range(CFG.N_FOLDS):
        train_df = df[df.fold!=fold]
        val_df = df[df.fold==fold]
        train_fn(train_df, fold, to_idx, co_idx,topics_embedding_cudf, contents_embedding_cudf, CFG.SAVE_DIR / f'fold_{fold}.pkl')
        val_df['predictions_proba'] = inference_stage2(val_df, to_idx, co_idx, topics_embedding_cudf, contents_embedding_cudf, CFG.SAVE_DIR / f'fold_{fold}.pkl')
        target_df = correlations_df[correlations_df.topic_id.isin(val_df.topic_id)].reset_index(drop=True)
        val_df = postprocess(val_df, threshold=CFG.THRESHOLD, top_n = 5)
        score = calc_f2_score_mean(target_df, val_df)
        LOGGER.info(f"threshold:{CFG.THRESHOLD} score:{score}")
        scores.append(score)
    LOGGER.info(f'CV score:{sum(scores)/len(scores)}')
    
if __name__ == '__main__':
    main()
    