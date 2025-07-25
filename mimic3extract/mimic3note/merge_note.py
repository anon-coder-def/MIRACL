import os
import argparse
import pandas as pd
import random
random.seed(42)
from tqdm import tqdm

import os
import pandas as pd
from mimic3note.utils import *


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_multimodal_listfile(args, phase):
    
    

    # 路径设定
    task_path = os.path.join(BASE_DIR, '../', args.root_path, args.task)
    note_path = os.path.join(BASE_DIR, '../', args.root_path, 'notes', f'mimic3_{phase}_notes.csv')
    listfile_path = os.path.join(task_path, f'{phase}_listfile.csv')
    output_path = os.path.join(task_path, f'{phase}_multimodal_listfile.csv')
    
    print("Expected file path:", os.path.abspath(listfile_path))

    # 读取 listfile 和 note file
    listfile = pd.read_csv(listfile_path)
    notes = pd.read_csv(note_path)

    # 标准化列名
    # listfile.columns = listfile.columns.str.lower()
    # notes.columns = notes.columns.str.upper()
    
    print(notes.columns)

    # 按 Recordtime 排序
    notes = notes.sort_values(by=['PatientID', 'ICUSTAY_ID', 'Recordtime'])

    # 合并前5条 note
    def get_merged_notes(patient_id, stay_id):
        filtered = notes[(notes['PatientID'] == patient_id) & (notes['ICUSTAY_ID'] == stay_id)]
        selected = filtered.head(5)
        merged_text = ' '.join(selected['Text'].astype(str).tolist())
        return merged_text

    # 生成 notes 列
    listfile['notes'] = listfile.apply(
        lambda row: get_merged_notes(row['patient_id'], row['stay_id']),
        axis=1
    )
    
    # Extract Brief Hospital Course part as the overall admission summary (For DRG task)
    listfile['notes'] = listfile['notes'].apply(remove_symbol_strict)
    
    
    # Swap notes to one index after period_length
    
    cols = list(listfile.columns)
    cols.remove('notes')
    idx = cols.index('period_length')
    cols.insert(idx+1, 'notes')
    listfile = listfile[cols]
    
    # listfile['brief_hospital_course'] = listfile['notes'].apply(extract_BHC)

    # # Extract Past Medical History  (For 6 tasks)
    # listfile['past_medical_history'] = listfile['notes'].apply(extract_PMH)
    
    
    
    # listfile.drop(['notes'], axis=1, inplace=True)

    # 保存输出
    listfile.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


    

def main():
    parser = argparse.ArgumentParser(description="Extract multimodal data for patients.")
    parser.add_argument('task', type=str, help='Task for MIMIC-III')
    # parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-IV CSV files.')
    parser.add_argument('root_path', type=str, help="Path to root folder patient information.")
    # parser.add_argument('ehr_path', type=str, help="Directory where the time series data should be stored.")
    # parser.add_argument('note_path', type=str, help="Directory where the note data should be stored.")
    args, _ = parser.parse_known_args()

    # if not os.path.exists(args.ehr_path):
    #     os.makedirs(args.ehr_path)


    # 执行两个阶段
    for split in ['train', 'test']:
        create_multimodal_listfile(args, split)


if __name__ == '__main__':
    main()
