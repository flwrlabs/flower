import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
    
class Generator():
    def __init__(self,cohort_output,if_mort,if_admn,if_los,feat_cond,feat_lab,feat_proc,feat_med,impute,include_time=24,bucket=1,predW=0):
        self.impute=impute
        self.feat_cond,self.feat_proc,self.feat_med,self.feat_lab = feat_cond,feat_proc,feat_med,feat_lab
        self.cohort_output=cohort_output
        
        self.data = self.generate_adm()
        print("[ READ COHORT ]")
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        if if_mort:
            print(predW)
            self.mortality_length(include_time,predW)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        print("done")
        self.smooth_meds(bucket)
        
        #if(self.feat_lab):
        #    print("[ ======READING LABS ]")
        #    nhid=len(self.hids)
        #    for n in range(0,nhids,10000):
        #        self.generate_labs(self.hids[n,n+10000])
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()
        if(self.feat_lab):
            print("[ ======READING LABS ]")
            self.generate_labs()
        
            
    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['admittime'] = pd.to_datetime(data['admittime'])
        data['dischtime'] = pd.to_datetime(data['dischtime'])
        data['los']=pd.to_timedelta(data['dischtime']-data['admittime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy', 'hours']] = data['los'].str.split(' ', expand=True)
        data[['hours', 'min', 'sec']] = data['hours'].str.split(':', expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)
        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['hadm_id'].isin(self.data['hadm_id'])]
        cond_per_adm = cond.groupby('hadm_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['hadm_id'].isin(self.data['hadm_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['proc_time_from_admit'].str.split(' ', expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]
        
        ###Remove where event time is after discharge time
        proc=pd.merge(proc,self.data[['hadm_id','los']],on='hadm_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']
        
        self.proc=proc
        
    def generate_labs(self):
        chunksize = 10000000
        final=pd.DataFrame()
        for labs in tqdm(pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            labs=labs[labs['hadm_id'].isin(self.data['hadm_id'])]
            labs[['start_days', 'dummy','start_hours']] = labs['lab_time_from_admit'].str.split(' ', expand=True)
            labs[['start_hours','min','sec']] = labs['start_hours'].str.split(':', expand=True)
            labs['start_time']=pd.to_numeric(labs['start_days'])*24+pd.to_numeric(labs['start_hours'])
            labs=labs.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
            labs=labs[labs['start_time']>=0]

            ###Remove where event time is after discharge time
            labs=pd.merge(labs,self.data[['hadm_id','los']],on='hadm_id',how='left')
            labs['sanity']=labs['los']-labs['start_time']
            labs=labs[labs['sanity']>0]
            del labs['sanity']
            
            if final.empty:
                final=labs
            else:
                final = pd.concat([final, labs], ignore_index=True)

        self.labs=final
        
    def generate_meds(self):
        meds=pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['hadm_id'].isin(self.data['hadm_id'])]
        meds=pd.merge(meds,self.data[['hadm_id','los']],on='hadm_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        
        meds['dose_val_rx']=meds['dose_val_rx'].apply(pd.to_numeric, errors='coerce')
        
        
        self.meds=meds
        
    
    def mortality_length(self,include_time,predW):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time+predW)]
        self.hids=self.data['hadm_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        
        self.data['los']=include_time
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###LAB
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=self.labs[self.labs['start_time']<=include_time]
            
        
        self.los=include_time
        
    def los_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['hadm_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        
        self.data['los']=include_time
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###LAB
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=self.labs[self.labs['start_time']<=include_time]
            
        
        #self.los=include_time    
    
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['hadm_id'].unique()
        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.data['hadm_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['hadm_id'].isin(self.data['hadm_id'])]
            self.meds=pd.merge(self.meds,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['hadm_id'].isin(self.data['hadm_id'])]
            self.proc=pd.merge(self.proc,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
        
        ###LABS
        if(self.feat_lab):
            self.labs=self.labs[self.labs['hadm_id'].isin(self.data['hadm_id'])]
            self.labs=pd.merge(self.labs,self.data[['hadm_id','select_time']],on='hadm_id',how='left')
            self.labs['start_time']=self.labs['start_time']-self.labs['select_time']
            self.labs=self.labs[self.labs['start_time']>=0]

            
    def smooth_meds(self,bucket):
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_labs=pd.DataFrame()
        
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        
        t=0
        for i in tqdm(range(0,self.los,bucket)):
             
            ###MEDS
            if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['hadm_id','drug_name']).agg({'stop_time':'max','subject_id':'max','dose_val_rx':np.nanmean})
                sub_meds=sub_meds.reset_index()
                sub_meds['start_time']=t
                sub_meds['stop_time']=sub_meds['stop_time']/bucket
                if final_meds.empty:
                    final_meds=sub_meds
                else:
                    final_meds= pd.concat([final_meds, sub_meds], ignore_index=True)
            
            ###PROC
            if(self.feat_proc):
                sub_proc=self.proc[(self.proc['start_time']>=i) & (self.proc['start_time']<i+bucket)].groupby(['hadm_id','icd_code']).agg({'subject_id':'max'})
                sub_proc=sub_proc.reset_index()
                sub_proc['start_time']=t
                if final_proc.empty:
                    final_proc=sub_proc
                else:    
                    final_proc= pd.concat([final_proc , sub_proc], ignore_index=True)
                    
            ###LABS
            if(self.feat_lab):
                sub_labs=self.labs[(self.labs['start_time']>=i) & (self.labs['start_time']<i+bucket)].groupby(['hadm_id','itemid']).agg({'subject_id':'max','valuenum':np.nanmean})
                sub_labs=sub_labs.reset_index()
                sub_labs['start_time']=t
                if final_labs.empty:
                    final_labs=sub_labs
                else:    
                    final_labs= pd.concat([final_labs , sub_labs], ignore_index=True)
            
            t=t+1
        los=int(self.los/bucket)
        
        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['hadm_id','drug_name']).size()
            self.med_per_adm=f2_meds.groupby('hadm_id').sum().reset_index()[0].max()        
            self.medlength_per_adm=final_meds.groupby('hadm_id').size().max()
        
        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['hadm_id','icd_code']).size()
            self.proc_per_adm=f2_proc.groupby('hadm_id').sum().reset_index()[0].max()        
            self.proclength_per_adm=final_proc.groupby('hadm_id').size().max()
            
        ###LABS
        if(self.feat_lab):
            f2_labs=final_labs.groupby(['hadm_id','itemid']).size()
            self.labs_per_adm=f2_labs.groupby('hadm_id').sum().reset_index()[0].max()        
            self.labslength_per_adm=final_labs.groupby('hadm_id').size().max()

        ###CREATE DICT
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        self.create_Dict(final_meds,final_proc,final_labs,los)
        
        
    def create_Dict(self, meds, proc, labs, los, chunk_size=5000):

        print("[creating data dictionaries]")

        datadic = {}
        time_index = pd.Index(range(los))

        csv_dir = "./data/csv"
        os.makedirs(csv_dir, exist_ok=True)
        dyn_path = os.path.join(csv_dir, "all_dynamic.csv")
        demo_path = os.path.join(csv_dir, "all_demo.csv")
        static_path = os.path.join(csv_dir, "all_static.csv")
        label_path = os.path.join(csv_dir, "labels.csv")

        # remove previous outputs
        for p in [dyn_path, demo_path, static_path, label_path]:
            if os.path.exists(p): os.remove(p)

        # Temporary storage for the current chunk
        chunk_demo, chunk_dyn, chunk_static = [], [], []
        
        print("saving labels.csv...")
        # ---------- labels ----------
        labels_df = self.data[['hadm_id','label']].drop_duplicates()
        labels_df.to_csv(label_path,index=False)

        # ---------- pre-group data (major speedup) ----------
        print("pre-grouping data...")
        med_groups  = dict(tuple(meds.groupby("hadm_id"))) if self.feat_med else {}
        proc_groups = dict(tuple(proc.groupby("hadm_id"))) if self.feat_proc else {}
        lab_groups  = dict(tuple(labs.groupby("hadm_id"))) if self.feat_lab else {}
        cond_groups = dict(tuple(self.cond.groupby("hadm_id"))) if self.feat_cond else {}
        data_groups = dict(tuple(self.data.groupby("hadm_id")))

        med_feat  = meds['drug_name'].unique() if self.feat_med else []
        proc_feat = proc['icd_code'].unique() if self.feat_proc else []
        lab_feat  = labs['itemid'].unique() if self.feat_lab else []
        cond_feat = self.cond['new_icd_code'].unique() if self.feat_cond else []

        def flush_to_disk(lst, path):
            if not lst:
                return
            df = pd.concat(lst)
            new = not os.path.exists(path)
            df.to_csv(path, mode="a", index=False, header=new)
            lst.clear()

        # ---------- main loop ----------
        print(f"Processing {len(self.hids)} patients...")
        for i, hid in enumerate(tqdm(self.hids)):

            grp = data_groups.get(hid)
            if grp is None:
                continue

            datadic[hid] = {
                'cond':{}, 'proc':{}, 'med':{}, 'lab':{},
                'ethnicity':grp['ethnicity'].iloc[0],
                'age':int(grp['Age'].iloc[0]),
                'gender':grp['gender'].iloc[0],
                'label':int(grp['label'].iloc[0])
            }

            # ---------- demo ----------
            demo = grp[['Age','gender','ethnicity','insurance']].copy()
            demo['hadm_id'] = hid
            chunk_demo.append(demo)

            patient_dyn = []

            # ---------- meds ----------
            if self.feat_med:
                m = med_groups.get(hid)

                if m is None:
                    val = pd.DataFrame(np.zeros((los,len(med_feat))),columns=med_feat)
                else:

                    val = m.pivot_table(index='start_time',
                                        columns='drug_name',
                                        values='dose_val_rx').reindex(time_index)

                    stop = m.pivot_table(index='start_time',
                                            columns='drug_name',
                                            values='stop_time').reindex(time_index)

                    stop = stop.ffill().fillna(0)
                    val = val.ffill().fillna(-1)

                    sig = (stop.sub(stop.index,axis=0) > 0).astype(int)

                    datadic[hid]['Med'] = {
                        'signal': sig.to_dict("list"),
                        'val': (sig*val).to_dict("list")
                    }

                    val = (sig*val).reindex(columns=med_feat,fill_value=0)

                val.columns = pd.MultiIndex.from_product([["MEDS"],val.columns])
                patient_dyn.append(val)

            # ---------- proc ----------
            if self.feat_proc:
                p = proc_groups.get(hid)

                if p is None:
                    df2 = pd.DataFrame(np.zeros((los,len(proc_feat))),columns=proc_feat)
                else:
                    p = p.copy()
                    p['val'] = 1

                    df2 = p.pivot_table(index='start_time',
                                        columns='icd_code',
                                        values='val').reindex(time_index)

                    df2 = (df2.fillna(0) > 0).astype(int)

                    datadic[hid]['proc'] = df2.to_dict("list")

                    df2 = df2.reindex(columns=proc_feat,fill_value=0)

                df2.columns = pd.MultiIndex.from_product([["proc"],df2.columns])
                patient_dyn.append(df2)

            # ---------- lab ----------
            if self.feat_lab:
                l = lab_groups.get(hid)

                if l is None:
                    val = pd.DataFrame(np.zeros((los,len(lab_feat))),columns=lab_feat)
                else:

                    val = l.pivot_table(index='start_time',
                                        columns='itemid',
                                        values='valuenum').reindex(time_index)

                    sig = l.copy()
                    sig['val'] = 1

                    sig = sig.pivot_table(index='start_time',
                                            columns='itemid',
                                            values='val').reindex(time_index)

                    sig = (sig.fillna(0) > 0).astype(int)

                    if self.impute == 'mean':
                        val = val.ffill().bfill().fillna(val.mean())

                    elif self.impute == 'median':
                        val = val.ffill().bfill().fillna(val.median())

                    val = val.fillna(0)

                    datadic[hid]['lab'] = {
                        'signal': sig.to_dict("list"),
                        'val': val.to_dict("list")
                    }

                    val = val.reindex(columns=lab_feat,fill_value=0)

                val.columns = pd.MultiIndex.from_product([["lab"],val.columns])
                patient_dyn.append(val)

            if patient_dyn:
                dyn_df = pd.concat(patient_dyn,axis=1)
                dyn_df[('hadm_id','hadm_id')] = hid
                chunk_dyn.append(dyn_df)

            # ---------- static (cond) ----------
            if self.feat_cond:
                cg = cond_groups.get(hid)

                if cg is None:
                    datadic[hid]['cond']={'fids':['<pad>']}
                    st = pd.DataFrame(np.zeros((1,len(cond_feat))),columns=cond_feat)

                else:
                    datadic[hid]['cond']={'fids':list(cg['new_icd_code'])}

                    cg = cg.copy()
                    cg['val'] = 1

                    st = cg.drop_duplicates().pivot(index='hadm_id',
                                                    columns='new_icd_code',
                                                    values='val')

                    st = st.reindex(columns=cond_feat,fill_value=0)

                st.columns = pd.MultiIndex.from_product([["cond"],st.columns])
                st['hadm_id'] = hid
                chunk_static.append(st)

            # ---------- flush ----------
            if (i+1) % chunk_size == 0:
                flush_to_disk(chunk_demo,demo_path)
                flush_to_disk(chunk_dyn,dyn_path)
                flush_to_disk(chunk_static,static_path)

        flush_to_disk(chunk_demo,demo_path)
        flush_to_disk(chunk_dyn,dyn_path)
        flush_to_disk(chunk_static,static_path)

        ###### SAVE DICTIONARIES ######
        os.makedirs("./data/dict", exist_ok=True)

        metaDic = {'Cond':{}, 'Proc':{}, 'Med':{}, 'Lab':{}, 'LOS':{}}
        metaDic['LOS'] = los

        # Core dictionaries
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(datadic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)


        # ---- DEMOGRAPHIC VOCABS ----
        eth_vocab_list = list(self.data['ethnicity'].unique())
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(eth_vocab_list, fp)
        self.eth_vocab = len(eth_vocab_list)

        age_vocab_list = list(self.data['Age'].unique())
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(age_vocab_list, fp)
        self.age_vocab = len(age_vocab_list)

        ins_vocab_list = list(self.data['insurance'].unique())
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(ins_vocab_list, fp)
        self.ins_vocab = len(ins_vocab_list)


        # ---- MEDICATION VOCAB ----
        if self.feat_med:
            med_vocab_list = list(meds['drug_name'].unique())
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(med_vocab_list, fp)

            self.med_vocab = len(med_vocab_list)
            metaDic['Med'] = self.med_per_adm


        # ---- CONDITION VOCAB ----
        if self.feat_cond:
            cond_vocab_list = list(self.cond['new_icd_code'].unique())
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(cond_vocab_list, fp)

            self.cond_vocab = len(cond_vocab_list)
            metaDic['Cond'] = self.cond_per_adm


        # ---- PROCEDURE VOCAB ----
        if self.feat_proc:
            proc_vocab_list = list(proc['icd_code'].unique())
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(proc_vocab_list, fp)

            self.proc_vocab = len(proc_vocab_list)
            metaDic['Proc'] = self.proc_per_adm


        # ---- LAB VOCAB ----
        if self.feat_lab:
            lab_vocab_list = list(labs['itemid'].unique())
            with open("./data/dict/labsVocab", 'wb') as fp:
                pickle.dump(lab_vocab_list, fp)

            self.lab_vocab = len(lab_vocab_list)
            metaDic['Lab'] = self.labs_per_adm


        # ---- META DICTIONARY ----
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
