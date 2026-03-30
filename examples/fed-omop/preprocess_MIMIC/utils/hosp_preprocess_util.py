import numpy as np
import pandas as pd
import os
import re
import ast
import datetime as dt
from tqdm import tqdm
from utils.labs_preprocess_util import *

from sklearn.preprocessing import MultiLabelBinarizer

########################## GENERAL ##########################
def dataframe_from_csv(path, compression='gzip', header=0, index_col=0, chunksize=None):
    return pd.read_csv(path, compression=compression, header=header, index_col=index_col, chunksize=chunksize)

def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'core/admissions.csv.gz'))
    admits=admits.reset_index()
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'ethnicity']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits


def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'core/patients.csv.gz'))
    pats = pats.reset_index()
    pats = pats[['subject_id', 'gender','dod','anchor_age','anchor_year', 'anchor_year_group']]
    pats['yob']= pats['anchor_year'] - pats['anchor_age']
    #pats.dob = pd.to_datetime(pats.dob)
    pats.dod = pd.to_datetime(pats.dod)
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(mimic4_path):
    diag = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/diagnoses_icd.csv.gz'))
    diag.reset_index(inplace=True)
    return diag


def read_d_icd_diagnoses_table(mimic4_path):
    d_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_diagnoses.csv.gz'))
    d_icd.reset_index(inplace=True)
    return d_icd[['icd_code', 'long_title']]


def read_diagnoses(mimic4_path):
    return read_diagnoses_icd_table(mimic4_path).merge(
        read_d_icd_diagnoses_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


def standardize_icd(mapping, df, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

    def icd_9to10(icd):
        # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except:
            print("Error on code", icd)
            return np.nan

    # Create new column with original codes as default
    col_name = 'icd10_convert'
    if root: col_name = 'root_' + col_name
    df[col_name] = df['icd_code'].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    for code, group in df.loc[df.icd_version == 9].groupby(by='icd_code'):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code


########################## LABS ##########################
def read_labevents_table(mimic4_path):
    labevents = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/labevents.csv.gz'), chunksize=1000)
    labevents.reset_index(inplace=True)
    return labevents[['subject_id', 'itemid', 'hadm_id', 'charttime', 'storetime', 'value', 'valueuom', 'flag']]


def read_d_labitems_table(mimic4_path):
    labitems = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_labitems.csv.gz'), chunksize=1000)
    labitems.reset_index(inplace=True)
    return labitems[['itemid', 'label', 'category', 'lonic_code']]


def read_labs(mimic4_path):
    labevents = read_labevents_table(mimic4_path)
    labitems =  read_d_labitems_table(mimic4_path)
    return labevents(mimic4_path).merge(
        labitems, how='inner', left_on=['itemid'], right_on=['itemid']
    )


########################## PROCEDURES ##########################
def read_procedures_icd_table(mimic4_path):
    proc = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/procedures_icd.csv.gz'))
    proc.reset_index(inplace=True)
    return proc


def read_d_icd_procedures_table(mimic4_path):
    p_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_procedures.csv.gz'))
    p_icd.reset_index(inplace=True)
    return p_icd[['icd_code', 'long_title']]


def read_procedures(mimic4_path):
    return read_procedures_icd_table(mimic4_path).merge(
        read_d_icd_procedures_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


########################## MEDICATIONS ##########################
def read_prescriptions_table(mimic4_path):
    meds = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/prescriptions.csv.gz'))
    meds = meds.reset_index()
    return meds[['subject_id', 'hadm_id', 'starttime', 'stoptime', 'ndc', 'gsn', 'drug', 'drug_type']]

def get_generic_drugs(mapping, df):
    """Takes NDC product table and prescriptions dataframe; adds column with NDC table's corresponding generic name"""

    def brand_to_generic(ndc):
        # We only want the first 2 sections of the NDC code: xxxx-xxxx-xx
        matches = list(re.finditer(r"-", ndc))
        if len(matches) > 1:
            ndc = ndc[:matches[1].start()]
        try:
            return mapping.loc[mapping.PRODUCTNDC == ndc].NONPROPRIETARYNAME.iloc[0]
        except:
            print("Error: ", ndc)
            return np.nan

    df['generic_drug_name'] = df['ndc'].apply(brand_to_generic)


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    mapping = pd.read_csv(map_path, header=0, delimiter='\t')
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def read_ndc_mapping(map_path):
    ndc_map = pd.read_csv(map_path, header=0, delimiter='\t')
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
    ndc_map.columns = list(map(str.lower, ndc_map.columns))
    return ndc_map


########################## PREPROCESSING ##########################
def get_range(df: pd.DataFrame, time_col:str, anchor_col:str, measure='days') -> pd.Series:
    """Uses array arithmetic to find the ranges an observation time could be in based on the patient's anchor info"""

    def get_year_timedelta(group: tuple) -> str:
        """Applied to a Series of tuples in the form (start year, end year) that represent the range of years a labevent could occur according to a patient's anchor info"""
        # Recall that we only want info in a 3 year time range anywhere between 2008 and 2019.
        # All possible ranges are:  ['2008 - 2010', '2009 - 2011', '2010 - 2012', '2011 - 2013',
        # '2012 - 2014', '2013 - 2015','2014 - 2016',  '2015 - 2017', '2016 - 2018',  '2017 - 2019']
        if group[0] >= 2008 and group[0] <= 2017 and group[1] >= 2010 and group[1] <= 2019:
            return group[0] - 2008
        else:
            return np.nan

    if measure == 'years':
        # Array arithmetic to obtain a tuple of (start year, end year) for each labevent
        # After getting list of tuples, apply get_timedelta
        shift = (df[time_col] - df[anchor_col])
        return pd.Series(list(zip(df.min_year_group + shift, df.max_year_group + shift))).apply(get_year_timedelta)
    elif measure =='days':
        # Convert base_year into a datetime and find the difference in days between the time_col and the beginning of the base_year
        base_dt = df[anchor_col].apply(lambda x: dt.datetime(year=x, month=1, day=1))
        return (df[time_col] - base_dt).dt.days
    else:
        raise Exception('\'measure\' argument must be either \'years\' or \'days\'.')

def preproc_meds(module_path:str, adm_cohort_path:str, mapping:str) -> pd.DataFrame:
  
    adm = pd.read_csv(adm_cohort_path, usecols=['hadm_id', 'admittime'], parse_dates = ['admittime'])
    med = pd.read_csv(module_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'drug', 'starttime', 'stoptime','ndc','dose_val_rx'], parse_dates = ['starttime', 'stoptime'])
    med = med.merge(adm, left_on = 'hadm_id', right_on = 'hadm_id', how = 'inner')
    med['start_hours_from_admit'] = med['starttime'] - med['admittime']
    med['stop_hours_from_admit'] = med['stoptime'] - med['admittime']
    
    # Normalize drug strings and remove potential duplicates

    med.drug = med.drug.fillna("").astype(str)
    med.drug = med.drug.apply(lambda x: x.lower().strip().replace(" ", "_") if not "" else "")
    med.drug=med.drug.dropna().apply(lambda x: x.lower().strip())
    
    #meds.to_csv(output_path, compression='gzip', index=False)
    med = ndc_meds(med,mapping)
    
    print("Number of unique type of drug: ", med.drug.nunique())
    print("Number of unique type of drug (after grouping to use Non propietary names): ", med.nonproprietaryname.nunique())
    print("Total number of rows: ", med.shape[0])
    print("# Admissions:  ", med.hadm_id.nunique())
    
    return med
    
    
def ndc_meds(med, mapping:str) -> pd.DataFrame:
    
    # Convert any nan values to a dummy value
    med.ndc = med.ndc.fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med.ndc = med.ndc.astype("Int64")
    
    # The NDC codes in the prescription dataset is the 11-digit NDC code, although codes are missing
    # their leading 0's because the column was interpreted as a float then integer; this function restores
    # the leading 0's, then obtains only the PRODUCT and MANUFACTUERER parts of the NDC code (first 9 digits)
    def to_str(ndc):
        if ndc < 0:         # dummy values are < 0
            return np.nan
        ndc = str(ndc)
        return (("0"*(11 - len(ndc))) + ndc)[0:-2]

    # The mapping table is ALSO incorrectly formatted for 11 digit NDC codes. An 11 digit NDC is in the
    # form of xxxxx-xxxx-xx for manufactuerer-product-dosage. The hyphens are in the correct spots, but
    # the number of digits within each section may not be 5-4-2, in which case we add leading 0's to each
    # to restore the 11 digit format. However, we only take the 5-4 sections, just like the to_str function
    def format_ndc_table(ndc):
        parts = ndc.split("-")
        return ("0"*(5 - len(parts[0])) + parts[0]) + ("0"*(4 - len(parts[1])) + parts[1])
    
    def read_ndc_mapping2(map_path):
        ndc_map = pd.read_csv(map_path, header=0, delimiter='\t', encoding = 'latin1')
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
        ndc_map.columns = list(map(str.lower, ndc_map.columns))
        return ndc_map
    
    # Read in NDC mapping table
    ndc_map = read_ndc_mapping2(mapping)[['productndc', 'nonproprietaryname', 'pharm_classes']]
    
    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map['new_ndc'] = ndc_map.productndc.apply(format_ndc_table)
    ndc_map.drop_duplicates(subset=['new_ndc', 'nonproprietaryname'], inplace=True)
    med['new_ndc'] = med.ndc.apply(to_str)  
    
    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how='inner', left_on='new_ndc', right_on='new_ndc')
    
    # In NDC mapping table, the pharm_class col is structured as a text string, separating different pharm classes from eachother
    # This can be [PE], [EPC], and others, but we're interested in EPC. Luckily, between each commas, it states if a phrase is [EPC]
    # So, we just string split by commas and keep phrases containing "[EPC]"
    def get_EPC(s):
        """Gets the Established Pharmacologic Class (EPC) from the mapping table"""
        if type(s) != str:
            return np.nan
        words = s.split(",")
        return [x for x in words if "[EPC]" in x]
    
    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med['EPC'] = med.pharm_classes.apply(get_EPC)
    
    return med

def preproc_labs(dataset_path: str, version_path:str, cohort_path:str, time_col:str, anchor_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    
    usecols = ['itemid','subject_id','hadm_id','charttime','valuenum','valueuom']
    dtypes = {
        'itemid':'int64',
        'subject_id':'int64',
        # 'hadm_id':'int64',            # hadm_id type not defined because it contains NaN values
        # 'charttime':'datetime64[ns]', # used as an argument in 'parse_cols' in pd.read_csv
        'value':'object',
        'valuenum':'float64',
        'valueuom':'object',
        'flag':'object'
    }
    df_cohort=pd.DataFrame()
    cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['admittime'])
    adm = pd.read_csv("./"+version_path+"/hosp/admissions.csv.gz", header=0, index_col=None, compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'], parse_dates=['admittime', 'dischtime'])
        
    # read module w/ custom params
    chunksize = 10000000
    for chunk in tqdm(pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col],chunksize=chunksize)):
        #print(chunk.shape)
        #chunk.dropna(subset=['hadm_id'],inplace=True,axis=1)
        chunk=chunk.dropna(subset=['valuenum'])
        chunk['valueuom']=chunk['valueuom'].fillna(0)
        
        chunk=chunk[chunk['subject_id'].isin(cohort['subject_id'].unique())]
        #print(chunk['hadm_id'].isna().sum())
        chunkna=chunk[chunk['hadm_id'].isna()]
        chunk=chunk[chunk['hadm_id'].notnull()]
        chunkna = impute_hadm_ids(chunkna[['subject_id','hadm_id','itemid','charttime','valuenum','valueuom']].copy(), adm)
        del chunkna['hadm_id']
        chunkna=chunkna.rename(columns={'hadm_id_new':'hadm_id'})
        chunkna=chunkna[['subject_id','hadm_id','itemid','charttime','valuenum','valueuom']]
        chunk = pd.concat([chunk, chunkna], ignore_index=True)
        #print(chunk['hadm_id'].isna().sum())
         
        chunk = chunk.merge(cohort[['hadm_id', 'admittime','dischtime']], how='inner', left_on='hadm_id', right_on='hadm_id')
        #print(chunk.head())
        chunk['charttime']=pd.to_datetime(chunk['charttime'])
        chunk['lab_time_from_admit'] = chunk['charttime'] - chunk['admittime']
        #chunk['valuenum']=chunk['valuenum'].fillna(0)
        chunk=chunk.dropna()
        
        #print(chunk.shape)
        #print(chunk.head())
        if df_cohort.empty:
            df_cohort=chunk
        else:
            df_cohort = pd.concat([df_cohort, chunk], ignore_index=True)
    
    #labs = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()
    
    
    
    #print(df_cohort.shape)
    #adm = pd.read_csv("./mimic-iv-1.0/core/admissions.csv.gz", header=0, index_col=None, compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'], parse_dates=['admittime', 'dischtime'])
    # labs.to_csv(".data/long_format/labs/labs.csv.gz", compression="gzip", index=False)
    #print(adm.head())                  
    # Use imputation function to impute missing hadm_ids where possible
    #labs = impute_hadm_ids(labs[['subject_id','hadm_id','itemid','charttime']].copy(), adm)
    #print(labs.shape)
    
    #print(labs.shape)
    #labs=labs.rename_columns(columns={'hadm_id_new':'hadm_id'})     
    #print(labs.shape)
    #cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['admittime'])
    #df_cohort = labs.merge(cohort[['hadm_id', 'admittime','dischtime']], how='inner', left_on='hadm_id', right_on='hadm_id')
    
    #df_cohort['lab_time_from_admit'] = df_cohort['charttime'] - df_cohort['admittime']
    #df_cohort['valuenum']=df_cohort['valuenum'].fillna(0)
    #df_cohort=df_cohort.dropna()
    print("# Itemid: ", df_cohort.itemid.nunique())
    print("# Admissions: ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort
    
    
def preproc_proc(dataset_path: str, cohort_path:str, time_col:str, anchor_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()

        # Only consider values in our cohort
        cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['admittime'])
        
        #print(module.head())
        #print(cohort.head())

        # merge module and cohort
        return module.merge(cohort[['hadm_id', 'admittime','dischtime']], how='inner', left_on='hadm_id', right_on='hadm_id')

    df_cohort = merge_module_cohort()
    df_cohort['proc_time_from_admit'] = df_cohort['chartdate'] - df_cohort['admittime']
    df_cohort=df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique ICD9 Procedures:  ", df_cohort.loc[df_cohort.icd_version == 9].icd_code.dropna().nunique())
    print("# Unique ICD10 Procedures: ",df_cohort.loc[df_cohort.icd_version == 10].icd_code.dropna().nunique())

    print("\nValue counts of each ICD version:\n", df_cohort.icd_version.value_counts())
    print("# Admissions:  ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort

def preproc_icd_module(module_path:str, adm_cohort_path:str, icd_map_path=None, map_code_colname=None, only_icd10=True) -> pd.DataFrame:
    """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""    
    
    def get_module_cohort(module_path:str, cohort_path:str):
        module = pd.read_csv(module_path, compression='gzip', header=0)
        adm_cohort = pd.read_csv(adm_cohort_path, compression='gzip', header=0)
        #print(module.head())
        #print(adm_cohort.head())
        
        #adm_cohort = adm_cohort.loc[(adm_cohort.timedelta_years <= 6) & (~adm_cohort.timedelta_years.isna())]
        return module.merge(adm_cohort[['hadm_id', 'label']], how='inner', left_on='hadm_id', right_on='hadm_id')

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a modulenosis dataframe; adds column with converted ICD10 column"""
        
        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
            except:
                #print("Error on code", icd)
                return np.nan

        # Create new column with original codes as default
        col_name = 'icd10_convert'
        if root: col_name = 'root_' + col_name
        df[col_name] = df['icd_code'].values

        # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
        for code, group in tqdm(df.loc[df.icd_version == 9].groupby(by='icd_code')):
            new_code = icd_9to10(code)
            for idx in group.index.values:
                # Modify values of original df at the indexes in the groups
                df.at[idx, col_name] = new_code

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df['root'] = df[col_name].apply(lambda x: x[:3] if type(x) is str else np.nan)

    module = get_module_cohort(module_path, adm_cohort_path)
    #print(module.shape)
    #print(module['icd_code'].nunique())

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        #print(icd_map)
        standardize_icd(icd_map, module, root=True)
        print("# unique ICD-9 codes",module[module['icd_version']==9]['icd_code'].nunique())
        print("# unique ICD-10 codes",module[module['icd_version']==10]['icd_code'].nunique())
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10)",module['root_icd10_convert'].nunique())
        print("# unique ICD-10 codes (After clinical gruping ICD-10 codes)",module['root'].nunique())
        print("# Admissions:  ", module.hadm_id.nunique())
    return module


def pivot_cohort(df: pd.DataFrame, prefix: str, target_col:str, values='values', use_mlb=False, ohe=True, max_features=None):
    """Pivots long_format data into a multiindex array:
                                            || feature 1 || ... || feature n ||
        || subject_id || label || timedelta ||
    """
    aggfunc = np.mean
    pivot_df = df.dropna(subset=[target_col])

    if use_mlb:
        mlb = MultiLabelBinarizer()
        output = mlb.fit_transform(pivot_df[target_col].apply(ast.literal_eval))
        output = pd.DataFrame(output, columns=mlb.classes_)
        if max_features:
            top_features = output.sum().sort_values(ascending=False).index[:max_features]
            output = output[top_features]
        pivot_df = pd.concat([pivot_df[['subject_id', 'label', 'timedelta']].reset_index(drop=True), output], axis=1)
        pivot_df = pd.pivot_table(pivot_df, index=['subject_id', 'label', 'timedelta'], values=pivot_df.columns[3:], aggfunc=np.max)
    else:
        if max_features:
            top_features = pd.Series(pivot_df[['subject_id', target_col]].drop_duplicates()[target_col].value_counts().index[:max_features], name=target_col)
            pivot_df = pivot_df.merge(top_features, how='inner', left_on=target_col, right_on=target_col)
        if ohe:
            pivot_df = pd.concat([pivot_df.reset_index(drop=True), pd.Series(np.ones(pivot_df.shape[0], dtype=int), name='values')], axis=1)
            aggfunc = np.max
        pivot_df = pivot_df.pivot_table(index=['subject_id', 'label', 'timedelta'], columns=target_col, values=values, aggfunc=aggfunc)

    pivot_df.columns = [prefix + str(i) for i in pivot_df.columns]
    return pivot_df
