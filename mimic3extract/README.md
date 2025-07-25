## MIMIC-III multimodal data process
1. Extract listfile.csv
```python
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} newdata/root/
python -m mimic3benchmark.scripts.validate_events newdata/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects newdata/root/
python -m mimic3benchmark.scripts.split_train_and_test newdata/root/


# mkdir data/ehr yourself
python -m mimic3benchmark.scripts.create_in_hospital_mortality newdata/root/ newdata/in-hospital-mortality/ newdata/ehr/
python -m mimic3benchmark.scripts.create_readmission newdata/root/ newdata/readmission/ newdata/ehr/
python -m mimic3benchmark.scripts.create_decompensation newdata/root/ newdata/decompensation/ newdata/ehr/
python -m mimic3benchmark.scripts.create_new_phenotyping newdata/root/ newdata/phenotyping newdata/ehr/
```

2. Create and combine with clinical note to get similar multimodal-listfile as FlexCare did. NOTE: newdata is equivalent to data in MIMIC-III
```python
python -m mimic3note.extract_notes mimic-iii/ newdata/root/
python -m mimic3note.preprocess_notes newdata/root/ newdata/notes/
python -m mimic3note.merge_note phenotyping newdata
```
