import pandas as pd
import os
import random
import re
import string



# Remove symbols and line breaks from text
def remove_symbol(text):
    text = text.replace('\n', '')

    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')

    return text


def remove_symbol_strict(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\n\r]+', ' ', text)  # PostgreSQL 风格换行替换
    text = text.translate(str.maketrans('', '', string.punctuation))  # 去标点
    return text


# extract Brief Hospital Course in the discharge summary
def extract_BHC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern1 = re.compile(r"brief hospital course:(.*?)medications on admission", re.DOTALL)
    pattern2 = re.compile(r"brief Hospital Course:(.*?)discharge medications", re.DOTALL)

    if "brief hospital course:" in text:
        if re.search(pattern1, text):
            match = re.search(pattern1, text).group(1).strip()
        elif re.search(pattern2, text):
            match = re.search(pattern2, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Chief Complaint in the discharge summary
def extract_CC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"chief complaint:(.*?)major surgical or invasive procedure", re.DOTALL)

    if "chief complaint:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Past Medical History in the discharge summary
def extract_PMH(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"past medical history:(.*?)social history", re.DOTALL)

    if "past medical history:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Medications on Admission in the discharge summary
def extract_MA(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"medications on admission:(.*?)discharge medications", re.DOTALL)

    if "medications on admission:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match

