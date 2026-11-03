import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

df = pd.read_csv('data/drug_smiles.csv', header=None, names=['ID', 'SMILES'])

def get_brics_fragments(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            frags = BRICS.BRICSDecompose(mol)
            return ' ; '.join(list(frags))
    except:
        return None
    return None 

print("正在处理 BRICS 划分...")
df['BRICS_Fragments'] = df['SMILES'].apply(get_brics_fragments)

output_file = 'drug_brics_results_more.csv'
df.to_csv(output_file, index=False)
print(f"处理完成！结果已保存至: {output_file}")

# 预览一下
print(df.head())