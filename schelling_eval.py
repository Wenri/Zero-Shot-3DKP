import json
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import IO
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from kp_utils import SchellingDataset



class SchellingIO(IO):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        super(SchellingIO, self).__init__()

    # @staticmethod
    def loop_over_test_datasets(self):
        schelling_dataset = SchellingDataset()
        # schelling_dataset.transform_off_files_to_ply(self.save_mesh) # Just once when changing shapes
        yield from schelling_dataset

    def save_kps(self, mesh, kps_3d, class_title, mesh_id, semantic_id, postfix='keypts'):
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title, mesh_id, postfix='keypts'):
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed()
               for fname in save_dir.glob(f"{mesh_id}_*_keypts.ply")}
        return kps


if __name__ == '__main__':
    pass
