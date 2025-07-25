from scipy import stats
import os
import argparse
import pandas as pd
"""
Preprocess PubMed abstracts or MIMIC-III reports
"""
import re
import json

from nltk import sent_tokenize, word_tokenize

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()


def filter_for_first_hrs(dataframe, _days=2):
    min_time = dataframe.CHARTTIME.min()
    return dataframe[dataframe.CHARTTIME < min_time + pd.Timedelta(days=_days)]


def getText(t):
    return " ".join(list(preprocess_mimic(t)))


def getSentences(t):
    return list(preprocess_mimic(t))





# Preprocess NOTEVENTS
def get_discharge_summary(df_notevents):

    cond1 = (df_notevents.CATEGORY == 'Discharge summary')
    cond2 = (df_notevents.DESCRIPTION == 'Report')

    df_discharge_smmary = df_notevents[cond1&cond2]

    return df_discharge_smmary




def extract_notes(args, partition):
    df = pd.read_csv(os.path.join(args.mimic3_path, 'NOTEEVENTS.csv'))
    
    
    df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
    df.STORETIME = pd.to_datetime(df.STORETIME)
    
    # df = get_discharge_summary(df)
    

    df2 = df[df.SUBJECT_ID.notnull()]
    df2 = df2[df2.HADM_ID.notnull()]
    df2 = df2[df2.CHARTTIME.notnull()]
    df2 = df2[df2.TEXT.notnull()]
    

    df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]


    del df

    # df_filtered = df2.groupby('HADM_ID').apply(
    #    lambda x: filter_for_first_hrs(x, 2))
    # print(df_filtered.shape)
    print(df2.groupby('HADM_ID').count().describe())
    '''
    count  55926.000000  55926.000000  55926.000000
    mean      28.957283     28.957283     28.957283
    std       59.891679     59.891679     59.891679
    min        1.000000      1.000000      1.000000
    25%        5.000000      5.000000      5.000000
    50%       11.000000     11.000000     11.000000
    75%       27.000000     27.000000     27.000000
    max     1214.000000   1214.000000   1214.000000
    '''

    
    dataset_path = os.path.join(args.root_path, partition)
    
    
    all_files = os.listdir(dataset_path)
    all_folders = list(filter(lambda x: x.isdigit(), all_files))
    

    output_folder = os.path.join(args.root_path, f"{partition}_text_fixed")
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    suceed = 0
    failed = 0
    failed_exception = 0

    all_folders = all_folders

    sentence_lens = []
    hadm_id2index = {}

    for folder in all_folders:
        print(folder)
        try:
            patient_id = int(folder)
            sliced = df2[df2.SUBJECT_ID == patient_id]
            if sliced.shape[0] == 0:
                print("No notes for PATIENT_ID : {}".format(patient_id))
                failed += 1
                continue
            sliced.sort_values(by='CHARTTIME')

            # get the HADM_IDs from the stays.csv.
            stays_path = os.path.join(dataset_path, folder, 'stays.csv')
            stays_df = pd.read_csv(stays_path)
            hadm_ids = list(stays_df.HADM_ID.values)

            for ind, hid in enumerate(hadm_ids):
                hadm_id2index[str(hid)] = str(ind)

                sliced = sliced[sliced.HADM_ID == hid]
                #text = sliced.TEXT.str.cat(sep=' ')
                #text = "*****".join(list(preprocess_mimic(text)))
                data_json = {}
                for index, row in sliced.iterrows():
                    #f.write("%s\t%s\n" % (row['CHARTTIME'], getText(row['TEXT'])))
                    data_json["{}".format(row['CHARTTIME'])
                            ] = getSentences(row['TEXT'])

                with open(os.path.join(output_folder, folder + '_' + str(ind+1)), 'w') as f:
                    json.dump(data_json, f)

            suceed += 1
        except:
            import traceback
            traceback.print_exc()
            print("Failed with Exception FOR Patient ID: %s", folder)
            failed_exception += 1

    print("Sucessfully Completed: %d/%d" % (suceed, len(all_folders)))
    print("No Notes for Patients: %d/%d" % (failed, len(all_folders)))
    print("Failed with Exception: %d/%d" % (failed_exception, len(all_folders)))


    with open(os.path.join(output_folder, 'test_hadm_id2index'), 'w') as f:
        json.dump(hadm_id2index, f)


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    args, _ = parser.parse_known_args()
    
    extract_notes(args, 'train')
    extract_notes(args, 'test')

if __name__ == '__main__':
    main()
