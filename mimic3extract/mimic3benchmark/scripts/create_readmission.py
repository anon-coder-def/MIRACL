import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6):
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    ehr_output_dir = args.ehr_path

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Processing {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        ts_files = sorted([f for f in os.listdir(patient_folder) if "timeseries" in f])

        stay_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))
        stay_df = stay_df.sort_values(by="INTIME").reset_index(drop=True)

        for idx, ts_filename in enumerate(ts_files):
            episode_id = int(ts_filename.split("_")[0].replace("episode", ""))
            lb_filename = ts_filename.replace("_timeseries", "")
            label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            if label_df.empty:
                continue

            icustay = label_df.iloc[0]["Icustay"]
            los = 24.0 * label_df.iloc[0]["Length of Stay"]
            if pd.isnull(los):
                continue

            # 判断 readmission
            readmission = 0
            if idx < len(stay_df) - 1:
                outtime = pd.to_datetime(stay_df.loc[idx, 'OUTTIME'])
                next_intime = pd.to_datetime(stay_df.loc[idx + 1, 'INTIME'])
                if (next_intime - outtime).days <= 30:
                    readmission = 1
            elif idx == len(stay_df) - 1:
                if pd.notnull(stay_df.loc[idx, 'DEATHTIME']):
                    outtime = pd.to_datetime(stay_df.loc[idx, 'OUTTIME'])
                    deathtime = pd.to_datetime(stay_df.loc[idx, 'DEATHTIME'])
                    if (deathtime - outtime).days <= 30:
                        readmission = 1
                        
                        

            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lines = tsfile.readlines()
                header, lines = lines[0], lines[1:]
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, partition, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in lines:
                        outfile.write(line)
                        
                        
                # Write current timeseries csv to newdata/ehr as well
                with open(os.path.join(ehr_output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                        

            xy_pairs.append((patient, icustay, output_ts_filename, los, readmission))

    print("Number of samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    else:
        xy_pairs = sorted(xy_pairs)

    output_listfile_path = os.path.join(output_dir, f"{partition}_listfile.csv")
    with open(output_listfile_path, "w") as listfile:
        listfile.write('patient_id,stay_id,stay,period_length,y_true\n')
        for (patient_id, stay_id, stay_filename, period_length, y_true) in xy_pairs:
            listfile.write(f"{patient_id},{stay_id},{stay_filename},{period_length:.6f},{int(y_true)}\n")



def main():
    parser = argparse.ArgumentParser(description="Create readmission task dataset from MIMIC-III.")
    parser.add_argument("root_path", type=str, help="Path to root folder with train/test.")
    parser.add_argument("output_path", type=str, help="Directory to save output.")
    parser.add_argument('ehr_path', type=str, help="Directory where the created timeseires files should be stored.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    process_partition(args, "train")
    process_partition(args, "test")


if __name__ == '__main__':
    main()