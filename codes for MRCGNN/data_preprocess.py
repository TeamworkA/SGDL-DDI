import networkx as nx

from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader



from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy
from sklearn.feature_extraction.text import CountVectorizer


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)

def load_brics_features(drug_list_ordered, brics_path):
    df = pd.read_csv(brics_path, on_bad_lines='skip') 
    
    id_to_fragments = dict(zip(df['ID'], df['BRICS_Fragments']))
    
    corpus = []
    for drug_id in drug_list_ordered:
        if drug_id in id_to_fragments:
            frags = id_to_fragments[drug_id]
            if pd.isna(frags):
                corpus.append("")
            else:
                corpus.append(frags.replace(";", " "))
        else:
            corpus.append("") 
            
    vectorizer = CountVectorizer(token_pattern=r'\[.*?\][^ ]*') 
    X = vectorizer.fit_transform(corpus)
    
    brics_features = X.toarray()
    print(f"BRICS features shape: {brics_features.shape}")
    
    return brics_features, brics_features.shape[1]



class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:,2]
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):


        return  (self.entity1[index], self.entity2[index], self.relationtype[index])


def load_data(args, val_ratio=0.1, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data

    drug_list = []
    with open('data/drug_listxiao.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            drug_list.append(row[0])

    print(len(drug_list))

    zhongzi=args.zhongzi

    def loadtrainvaltest():
        #train dataset
        train=pd.read_csv('data/'+str(zhongzi)+'/ddi_training1xiao.csv')
        val = pd.read_csv('data/'+str(zhongzi)+'/ddi_validation1xiao.csv')
        test = pd.read_csv('data/'+str(zhongzi)+'/ddi_test1xiao.csv')

        max_type = max(train['type'].max(), val['type'].max(), test['type'].max())
        num_classes = int(max_type + 1)

        train_pos=[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(train_pos)
        train_pos = np.array(train_pos)
        for i in range(train_pos.shape[0]):
            train_pos[i][0] = int(drug_list.index(train_pos[i][0]))
            train_pos[i][1] = int(drug_list.index(train_pos[i][1]))
            train_pos[i][2] = int(train_pos[i][2])
        label_list=[]
        for i in range(train_pos.shape[0]):
            #label=np.zeros((65))
            label=np.zeros((num_classes))
            label[int(train_pos[i][2])]=1
            label_list.append(label)
        label_list=np.array(label_list)
        train_data= np.concatenate([train_pos, label_list],axis=1)

        #val dataset
        # val = pd.read_csv('data/'+str(zhongzi)+'/ddi_validation1.csv')
        val_pos = [(h, t, r) for h, t, r in zip(val['d1'], val['d2'], val['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(val_pos)
        val_pos= np.array(val_pos)
        for i in range(len(val_pos)):
            val_pos[i][0] = int(drug_list.index(val_pos[i][0]))
            val_pos[i][1] = int(drug_list.index(val_pos[i][1]))
            val_pos[i][2] = int(val_pos[i][2])
        label_list = []
        for i in range(val_pos.shape[0]):
            #label = np.zeros((65))
            label=np.zeros((num_classes))
            label[int(val_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        val_data = np.concatenate([val_pos, label_list], axis=1)

        #test dataset
        # test = pd.read_csv('data/'+str(zhongzi)+'/ddi_test1.csv')
        test_pos = [(h, t, r) for h, t, r in zip(test['d1'],test['d2'], test['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(test_pos)
        test_pos= np.array(test_pos)
        print(test_pos[0])
        for i in range(len(test_pos)):
            test_pos[i][0] = int(drug_list.index(test_pos[i][0]))
            test_pos[i][1] = int(drug_list.index(test_pos[i][1]))
            test_pos[i][2] = int(test_pos[i][2])
        label_list = []
        for i in range(len(test_pos)):
            #label = np.zeros((65))
            label=np.zeros((num_classes))
            label[int(test_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        test_data = np.concatenate([test_pos, label_list], axis=1)
        print(train_data.shape)
        print(val_data.shape)
        print(test_data.shape)
        return train_data,val_data,test_data

    train_data,val_data,test_data=loadtrainvaltest()
    params = {'batch_size': args.batch, 'shuffle': False, 'num_workers': args.workers, 'drop_last': False}


    train_types = train_data[:, 2].astype(int)
    val_types = val_data[:, 2].astype(int)
    test_types = test_data[:, 2].astype(int)

    max_type = int(max(train_types.max(), val_types.max(), test_types.max()))
    num_classes = max_type + 1
    print(f"Detected number of classes: {num_classes}")


    training_set = Data_class(train_data)

    train_loader = DataLoader(training_set, **params)


    validation_set = Data_class(val_data)

    val_loader = DataLoader(validation_set, **params)


    test_set = Data_class(test_data)

    test_loader = DataLoader(test_set, **params)

    # print('Extracting BRICS features...')
    brics_feat_np, brics_dim = load_brics_features(drug_list, 'data/drug_brics_results.csv')
    brics_features = torch.tensor(brics_feat_np, dtype=torch.float)

    # print('Extracting features...')



    features = np.load('feature/drug_emb_feature0.npy')
    ids = np.load('feature/drug_emb_feature0_drug_ids.npy', allow_pickle=True)
    ids=ids.tolist()
    features1=[]
    for i in range(len(drug_list)):
        features1.append(features[ids.index(drug_list[i])])

    features=np.array(features1)
    features_o = normalize(features)

    args.dimensions = features_o.shape[1]

    # adversarial nodes

    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]

    x_o = torch.tensor(features_o, dtype=torch.float)
    positive1 = copy.deepcopy(train_data)
    edge_index_o = []
    label_list = []

    for i in range(positive1.shape[0]):
        # 正向边
        edge_index_o.append([int(positive1[i][0]), int(positive1[i][1])])
        label_list.append(int(positive1[i][2]))
        # 反向边 (无向图)
        edge_index_o.append([int(positive1[i][1]), int(positive1[i][0])])
        label_list.append(int(positive1[i][2]))

    edge_index_o = torch.tensor(edge_index_o, dtype=torch.long)
    data_o = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)

    x_a = torch.tensor(features_a, dtype=torch.float)
    data_s = Data(x=x_a, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)

    print('Loading finished!')
    #return data_o, data_s, data_a, train_loader, val_loader, test_loader
    return data_o,data_s, train_loader, val_loader, test_loader,brics_features, brics_dim,num_classes
