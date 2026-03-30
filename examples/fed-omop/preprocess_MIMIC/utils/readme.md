### Files in the folder

- **hosp_preprocess_util.py** and **icu_preprocess_util.py**
  These files are used to read original feature csv files downloaded from MIMIC-IV and clean (removing NAs, removing duplicates, etc) and
  save feature files for the selected cohort in ./data/features folder.
  These files are run from **Block 2** in **mainPipeline.ipynb**
  
- **outlier_removal.py**
  removes outlier or imputes outlier with outlier threshold values.
  Used in **Block 6** in **mainPipeline.ipynb**
  
- **uom_conversion.py**
  unit conversion to highest frequency unit for each itemid in labevents and chartevents data
  Used as cleaning process in **Block 2** in **mainPipeline.ipynb**.


- **labs_preprocess_util.py**
  finds the missing admission ids in labevents data by placing timestamp of labevent  between the admission and discharge time of the admission for the patient.
  Used as cleaning process in **Block 2** in **mainPipeline.ipynb**
