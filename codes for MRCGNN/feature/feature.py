import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse

from blip2opt import mol_to_graph_data_obj_simple, Blip2OPT

def extract_and_save_embeddings(
    csv_path="data/drug_brics_results_more.csv", 
    save_path="drug_emb_feature_more0.npy",
    extract_mode="multimodal" 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = get_args()

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    drug_ids = df['ID'].values
    
    print("Converting SMILES to Graph representations...")
    graph_data_list = []
    for smiles in tqdm(df['SMILES']):
        graph = mol_to_graph_data_obj_simple(smiles)
        graph_data_list.append(graph)
        
    loader = DataLoader(graph_data_list, batch_size=32, shuffle=False)
    
    print("Initializing Blip2OPT Model...")
    model = Blip2OPT(
        bert_name="scibert",   
        gin_num_layers=5,                
        gin_hidden_dim=300,              
        gin_drop_ratio=0.2,              
        tune_gnn=False,
        opt_model="facebook/galactica-125m", 
        llm_tune="freeze",
        args=args,                       
        use_nas=args.NAS
    ).to(device)
    
    checkpoint_path = 'gin_pretrained/graphcl_80.pth' 
    print(f"Loading pretrained weights from {checkpoint_path}...")

    state_dict = torch.load(checkpoint_path, map_location=device)

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded with message: {msg}")
    
    model.eval()
    all_embeddings = []
    
    print(f"Extracting {extract_mode} embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            graph_embeds, graph_masks = model.graph_encoder(batch) 
            graph_embeds = model.ln_graph(graph_embeds)
            
            if extract_mode == "graph_only":
                emb = graph_embeds[:, 0, :] 
                
            elif extract_mode == "multimodal":
                query_tokens = model.query_tokens.expand(graph_embeds.shape[0], -1, -1)
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=graph_embeds,
                    encoder_attention_mask=graph_masks,
                    return_dict=True,
                )
                mol_tokens = model.opt_proj(query_output.last_hidden_state)
                
                emb = mol_tokens.mean(dim=1) 
            else:
                raise ValueError("Invalid extract_mode")

            all_embeddings.append(emb.cpu().numpy())
            
    
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Done! Extracted embeddings shape: {final_embeddings.shape}")
    
    
    np.save(save_path, final_embeddings)
    
    
    np.save(save_path.replace(".npy", "_drug_more_ids.npy"), drug_ids)
    print(f"Embeddings saved to {save_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--double', type=bool, default=False)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--valid_root', type=str, default='')
    parser.add_argument('--test_root', type=str, default='')
    parser.add_argument('--DDI', action="store_true", default=False)
    parser.add_argument('--drugbank', action="store_true", default=False)
    parser.add_argument('--NAS', action="store_true", default=False)
    parser.add_argument('--peft_config', type=str, default=None)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    args, unknown = parser.parse_known_args() 
    return args

if __name__ == '__main__':
    extract_and_save_embeddings(
        csv_path="data/drug_brics_results_more.csv",
        save_path="drug_emb_feature_more0.npy",
        extract_mode="multimodal" 
    )