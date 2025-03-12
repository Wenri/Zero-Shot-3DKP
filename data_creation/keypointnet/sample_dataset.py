import os
import json
import argparse
import numpy as np
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sampling shapes from keypointnet dataset.')
    parser.add_argument('--max_shapes_per_split', type=int, default=160,
                        help='Number of pairs per shape (default: 160)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the dataset. should be the same for the subsequent scripts')
    parser.add_argument('--keypointnet_dir', type=str, required=True,
                        help='Directory where the MVImgNet dataset is stored')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='List of dataset splits (default: ["train", "val", "test"])')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed for reproducibility (default: 2024)')
    return parser.parse_args()


def sample_split_shapes(dataframe, n_shapes_per_cat, random_seed):
    # This will group the DataFrame by the 'mesh_class' and sample 50 entries from each group
    return dataframe.groupby('mesh_class').apply(lambda x: x.sample(min(len(x), n_shapes_per_cat), random_state=random_seed)).reset_index(drop=True)

def sample_shapes(args, save=False):
    ret = {}
    
    for split in args.splits:
        ret[split] = []
        
        split_shapes_df = pd.read_csv(os.path.join(args.keypointnet_dir, 'splits', f'{split}.txt'), sep='-',
                                      names=('mesh_class', 'mesh_id'), dtype=str)
            
        n_cats = len(list(split_shapes_df.mesh_class.unique()))
        n_shapes_per_cat = args.max_shapes_per_split // n_cats
        
        ret[split] = sample_split_shapes(split_shapes_df, n_shapes_per_cat, args.seed)
        
    if save:
        os.makedirs(args.save_dir, exist_ok=True)  
        for split, sampled_shapes_df in ret.items():
            save_path = os.path.join(args.save_dir, f'{split}_shapes.csv')
            sampled_shapes_df.to_csv(save_path, index=False)
    
    return ret


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Read the training and the validation shapes
    _ = sample_shapes(args, save=True)