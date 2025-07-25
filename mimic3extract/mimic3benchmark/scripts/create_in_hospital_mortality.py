import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = args.output_path
    ehr_output_dir = args.ehr_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, partition, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                        
                # Write current timeseries csv to newdata/ehr as well
                with open(os.path.join(ehr_output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                
                
                        
                icustay = label_df['Icustay'].iloc[0]
                

                xy_pairs.append((patient, icustay, output_ts_filename, los, mortality))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    elif partition == "test":
        xy_pairs = sorted(xy_pairs)

    # 写入 listfile.csv 到对应的 partition 目录
    output_listfile_path = os.path.join(output_dir, f"{partition}_listfile.csv")
    with open(output_listfile_path, "w") as listfile:
        listfile.write('patient_id,stay_id,stay,period_length,y_true\n')
        for (patient_id, stay_id, stay_filename, period_length, y_true) in xy_pairs:
            listfile.write(f"{patient_id},{stay_id},{stay_filename},{period_length:.6f},{int(y_true)}\n")

    print(f"[{partition.upper()}] listfile.csv written to: {output_listfile_path}")


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('ehr_path', type=str, help="Directory where the created timeseires files should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
