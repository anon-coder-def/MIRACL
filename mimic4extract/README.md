## MIMIC-IV multimodal data process

1. Extract patient basic information from raw MIMIC-IV csv files and generate some labels for different tasks (Processing takes about 1 hour)
```python
python -m mimic4extract.scripts.extract_subjects_iv {PATH TO MIMIC-IV CSVs} data/root
```

2. Fix some issuses and removes the events that have missing information (Processing takes about 40 minutes)
```python
python -m mimic4extract.scripts.validate_events data/root
```

3. Extract time series and episode-level information (Processing takes about 2 hours)
```python
python -m mimic4extract.scripts.extract_episodes_from_subjects data/root
```

4. Extract multimodal data and divide patients into train/val/test
```python
python -m mimic4extract.scripts.extract_multimodal_data {PATH TO MIMIC-IV CSVs} {PATH TO MIMIC-NOTE} data/root/ data/ehr data/note
```

5. Create task-specific multimodal datasets and split into train/val/test sets
```python
python -m mimic4extract.scripts.create_phenotyping data/root data/ehr/ data/note data/phenotyping
python -m mimic4extract.scripts.create_diagnosis {PATH TO MIMIC-IV CSVs} data/root data/ehr/ data/note data/diagnosis
```

