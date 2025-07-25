import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split



class Multimodal_dataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, ehr_dir, task, return_names=True, period_length=48.0):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.period_length = period_length
        self.task = task

        self.ehr_dir = ehr_dir
        # self.note_dir = note_dir

        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[6:]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]
        
        
        self.data_map = {
            mas[1]+'_time_'+str(mas[3]): {
                'subject_id': float(mas[0]),
                'stay_id': float(mas[1]),
                'ehr_file': str(mas[2]),
                'time': str(mas[3]),
                'note': str(mas[4]),
                'labels': list(map(float, mas[5:])),
            }
            for mas in self._data
        }
        

        self.names = list(self.data_map.keys())

    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        with open(os.path.join(self.ehr_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return np.stack(ret), header


    def read_by_file_name(self, index, time_bound=None):
        ehr_file = self.data_map[index]['ehr_file']
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        t = float(t) if t!='' else -1
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        subject_id = self.data_map[index]['subject_id']
        note = self.data_map[index]['note']
        if self.task in ['decompensation', 'length-of-stay', 'diagnosis']:
            time_bound = t

        if ehr_file=='':
            (X, header) = None, None
        else:
            (X, header) = self._read_timeseries(ehr_file, time_bound=time_bound)


        return {"X": X,
                "t": t,
                "y": y,
                "note": note,
                'stay_id': stay_id,
                "header": header,
                "name": index,
                "subject_id": subject_id}

    def __getitem__(self, index, time_bound=None):
        id = index
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self.period_length
        note = ret["note"]
        ys = ret["y"]
        names = ret["name"]
        patient_ids = ret["subject_id"]
        
        # Check if noisy labels exist
        noisy_y = self.data_map[index].get('noisy_label', None)

        if data is not None:
            data = self.discretizer.transform(data, end=ts)[0]
            if self.normalizer is not None:
                data = self.normalizer.transform(data)
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        
        
        return data, note, ys, noisy_y, index, ret, self.task, id, patient_ids

    def __len__(self):
        return len(self.names)
    
    @property
    def y(self):
        """
        Return all labels for the dataset.
        """
        return [self.data_map[name]['labels'] for name in self.names]
    
    def y_noisy(self):
        """
        Return all labels for the dataset.
        """
        return [self.data_map[name]['noisy_label'] for name in self.names]
    
    def set_noisy_labels(self, noisy_labels):
        """
        Add noisy labels to the dataset.
        Args:
            noisy_labels (list): A list of noisy labels corresponding to the dataset instances.
        """
        assert len(noisy_labels) == len(self.names), "Noisy labels must match the dataset size."
        for name, noisy_label in zip(self.names, noisy_labels):
            self.data_map[name]['noisy_label'] = noisy_label
            
    
        


def get_imdb_multimodal_datasets(args):
    
    with h5py.File(f'{args.imdb_path}/multimodal_imdb.hdf5', 'r') as f:
        total_size = f['genres'].shape[0]
        indices = list(range(total_size))

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_ds = MultiModalIMDBDataset(f'{args.imdb_path}/multimodal_imdb.hdf5', indices=train_idx)
    test_ds = MultiModalIMDBDataset(f'{args.imdb_path}/multimodal_imdb.hdf5', indices=test_idx)
                
    return train_ds, test_ds


def get_multimodal_datasets(discretizer, normalizer, args, task):

    if args.subset:
        train_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/train_val_multimodal_listfile_subset.csv',
                          args.ehr_path, task)
        test_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/test_multimodal_listfile_subset.csv',
                         args.ehr_path, task)
    else:
        if args.dataset == 'mimic4':
            
            train_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/train_val_multimodal_listfile.csv',
                          args.ehr_path, task)
        
            test_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/test_multimodal_listfile.csv',
                         args.ehr_path, task)
            
        else:
            train_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/train_multimodal_listfile.csv',
                          args.ehr_path, task)
            
            test_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/test_multimodal_listfile.csv',
                         args.ehr_path, task)    
    
    return train_ds, test_ds



def get_mimic3_multimodal_datasets(discretizer, normalizer, args, task):
    train_transforms, test_transforms = get_transforms(args)

    train_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/train_multimodal_listfile.csv',
                        args.ehr_path, task)
    test_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/test_multimodal_listfile.csv',
                        args.ehr_path, task)
    

    return train_ds, test_ds
