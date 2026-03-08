import pandas as pd

def ADAS_filtering(adas_original_df, demographics_df):
	# Filter by PHASE
	adas_df=adas_original_df[adas_original_df['PHASE']=='ADNI3']
	# Filter by Subject ID
	adas_df = adas_df[adas_df['PTID'].astype(str).isin(demographics_df['Subject ID'].astype(str).unique())].copy()
	# Filter by Visit Code: bl, init
	valid_vis = {'bl', 'init'}
	mask = pd.Series(False, index=adas_df.index)
	for col in ('VISCODE', 'VISCODE2'):
		if col in adas_df.columns:
			mask |= adas_df[col].astype(str).str.lower().isin(valid_vis)
	adas_df = adas_df[mask].copy()
	# Check for duplicates
	duplicates = adas_df.duplicated(subset=['PTID'], keep=False)
	if duplicates.any():
		dup_ids = adas_df.loc[duplicates, 'PTID'].unique()
		for dup_id in dup_ids:
			dup_rows = adas_df[adas_df['PTID'] == dup_id]
			# Keep the row with the earliest visit (assuming VISCODE indicates time)
			if 'VISCODE' in dup_rows.columns:
				vis_order = dup_rows['VISCODE'].astype(str).str.lower().map({'bl': 0, 'init': 1}).fillna(2)
				earliest_row = dup_rows.loc[vis_order.idxmin()]
			else:
				earliest_row = dup_rows.iloc[0]
			# Drop other duplicates
			rows_to_drop = dup_rows.index.difference([earliest_row.name])
			adas_df = adas_df.drop(rows_to_drop)
	# Check missing subjects
	missing_subs = set(demographics_df['Subject ID'].astype(str).unique()) - set(adas_df['PTID'].astype(str).unique())
	if missing_subs:
		print(f"Warning: The following subjects are missing in ADAS data: {missing_subs}")
	
	# Get only relevant columns
	adas_df = adas_df[['PHASE', 'PTID', 'TOTSCORE', 'TOTAL13']]
	# Rename ADAS score columns if present
	adas_df = adas_df.rename(columns={'TOTSCORE': 'ADAS-Cog-11', 'TOTAL13': 'ADAS-Cog-13'})
	
	return adas_df

def CDR_filtering(CDR_original_df, demographics_df):
	
	# Filter by PHASE
	CDR_df=CDR_original_df[CDR_original_df['PHASE']=='ADNI3']
	# Filter by Subject ID
	CDR_df = CDR_df[CDR_df['PTID'].astype(str).isin(demographics_df['Subject ID'].astype(str).unique())].copy()
	# Filter by Visit Code: sc, init
	valid_vis = {'sc', 'init'}
	mask = pd.Series(False, index=CDR_df.index)
	for col in ('VISCODE', 'VISCODE2'):
		if col in CDR_df.columns:
			mask |= CDR_df[col].astype(str).str.lower().isin(valid_vis)
	CDR_df = CDR_df[mask].copy()
	# Check for duplicates
	duplicates = CDR_df.duplicated(subset=['PTID'], keep=False)
	if duplicates.any():
		dup_ids = CDR_df.loc[duplicates, 'PTID'].unique()
		for dup_id in dup_ids:
			dup_rows = CDR_df[CDR_df['PTID'] == dup_id]
			# Keep the row with the earliest visit (assuming VISCODE indicates time)
			if 'VISCODE' in dup_rows.columns:
				vis_order = dup_rows['VISCODE'].astype(str).str.lower().map({'sc': 0, 'init': 1}).fillna(2)
				earliest_row = dup_rows.loc[vis_order.idxmin()]
			else:
				earliest_row = dup_rows.iloc[0]
			# Drop other duplicates
			rows_to_drop = dup_rows.index.difference([earliest_row.name])
			CDR_df = CDR_df.drop(rows_to_drop)
	# Check missing subjects
	missing_subs = set(demographics_df['Subject ID'].astype(str).unique()) - set(CDR_df['PTID'].astype(str).unique())
	if missing_subs:
		print(f"Warning: The following subjects are missing in CDR data: {missing_subs}")
	# Get only relevant columns
	CDR_df = CDR_df[['PHASE', 'PTID', 'CDGLOBAL', 'CDRSB']]
	# Rename CDR score columns if present
	CDR_df = CDR_df.rename(columns={'CDGLOBAL': 'CDR-G', 'CDRSB': 'CDR-SB'})

	return CDR_df


def MoCA_filtering(MOCA_original_df, demographics_df):
	
	# Filter by PHASE
	MOCA_df=MOCA_original_df[MOCA_original_df['PHASE']=='ADNI3']
	# Filter by Subject ID
	MOCA_df = MOCA_df[MOCA_df['PTID'].astype(str).isin(demographics_df['Subject ID'].astype(str).unique())].copy()
	
	# Filter by Visit Code: bl, init
	valid_vis = {'bl', 'init'}
	mask = pd.Series(False, index=MOCA_df.index)
	for col in ('VISCODE', 'VISCODE2'):
		if col in MOCA_df.columns:
			mask |= MOCA_df[col].astype(str).str.lower().isin(valid_vis)
	MOCA_df = MOCA_df[mask].copy()
	# Check for duplicates
	duplicates = MOCA_df.duplicated(subset=['PTID'], keep=False)
	if duplicates.any():
		dup_ids = MOCA_df.loc[duplicates, 'PTID'].unique()
		for dup_id in dup_ids:
			dup_rows = MOCA_df[MOCA_df['PTID'] == dup_id]
			# Keep the row with the earliest visit (assuming VISCODE indicates time)
			if 'VISCODE' in dup_rows.columns:
				vis_order = dup_rows['VISCODE'].astype(str).str.lower().map({'bl': 0, 'init': 1}).fillna(2)
				earliest_row = dup_rows.loc[vis_order.idxmin()]
			else:
				earliest_row = dup_rows.iloc[0]
			# Drop other duplicates
			rows_to_drop = dup_rows.index.difference([earliest_row.name])
			MOCA_df = MOCA_df.drop(rows_to_drop)
	
	# Check missing subjects
	missing_subs = set(demographics_df['Subject ID'].astype(str).unique()) - set(MOCA_df['PTID'].astype(str).unique())
	if missing_subs:
		print(f"Warning: The following subjects are missing in MOCA data: {missing_subs}")
	
	# Operations from https://emilychenyh.github.io/ADNISurvival/preprocessing/cleaning/neuropsychological/moca/?utm_source=chatgpt.com
	MOCA_df['MOCAVISS'] = MOCA_df[['TRAILS','CUBE','CLOCKCON','CLOCKNO','CLOCKHAN']].sum(axis=1, skipna=True)
	MOCA_df['MOCANAME'] = MOCA_df[['LION','RHINO','CAMEL']].sum(axis=1, skipna=True)

	# MOCAATT:
	#  - digit span: sum of DIGFOR and DIGBACK
	#  - letters flag: 1 if LETTERS <= 1 else 0
	#  - serial flag: based on sum of SERIAL1..SERIAL5
	digits = MOCA_df[['DIGFOR','DIGBACK']].sum(axis=1, skipna=True)
	letters_flag = MOCA_df['LETTERS'].le(1).astype(int)  # NaN -> False -> 0

	serial_sum = MOCA_df[['SERIAL1','SERIAL2','SERIAL3','SERIAL4','SERIAL5']].sum(axis=1, skipna=True)
	serial_flag = pd.Series(0, index=MOCA_df.index, dtype=int)
	serial_flag = serial_flag.where(serial_sum.notna(), 0)  # keep 0 if all NaN
	serial_flag.loc[serial_sum == 0] = 0
	serial_flag.loc[serial_sum == 1] = 1
	serial_flag.loc[serial_sum.isin([2, 3])] = 2
	serial_flag.loc[serial_sum.isin([4, 5])] = 3

	MOCA_df['MOCAATT'] = digits + letters_flag + serial_flag

	MOCA_df['MOCALAN'] = MOCA_df[['REPEAT1','REPEAT2']].sum(axis=1, skipna=True) + MOCA_df['FFLUENCY'].ge(11).astype(int)
	MOCA_df['MOCAABS'] = MOCA_df[['ABSTRAN','ABSMEAS']].sum(axis=1, skipna=True)
	MOCA_df['MOCADEL'] = MOCA_df[['DELW1','DELW2','DELW3','DELW4','DELW5']].eq(1).sum(axis=1)
	MOCA_df['MOCAORI'] = MOCA_df[['DATE','MONTH','YEAR','DAY','PLACE','CITY']].sum(axis=1, skipna=True)
	# Get only relevant columns
	MOCA_df.columns = [col.replace('MOCA', 'MoCA') for col in MOCA_df.columns]
	MOCA_df = MOCA_df[['PHASE', 'PTID', 'MoCA', 'MOCAVISS', 'MOCANAME', 'MOCAATT', 'MOCALAN', 'MOCAABS', 'MOCADEL', 'MOCAORI']]
	
	return MOCA_df

def MMSE_filtering(MMSE_original_df, demographics_df):
	# Filter by PHASE
	MMSE_df=MMSE_original_df[MMSE_original_df['PHASE']=='ADNI3']
	# Filter by Subject ID
	MMSE_df = MMSE_df[MMSE_df['PTID'].astype(str).isin(demographics_df['Subject ID'].astype(str).unique())].copy()
	# Filter by Visit Code: sc, init
	valid_vis = {'sc', 'init'}
	mask = pd.Series(False, index=MMSE_df.index)
	for col in ('VISCODE', 'VISCODE2'):
		if col in MMSE_df.columns:
			mask |= MMSE_df[col].astype(str).str.lower().isin(valid_vis)
	MMSE_df = MMSE_df[mask].copy()
	# Check for duplicates
	duplicates = MMSE_df.duplicated(subset=['PTID'], keep=False)
	if duplicates.any():
		dup_ids = MMSE_df.loc[duplicates, 'PTID'].unique()
		for dup_id in dup_ids:
			dup_rows = MMSE_df[MMSE_df['PTID'] == dup_id]
			# Keep the row with the earliest visit (assuming VISCODE indicates time)
			if 'VISCODE' in dup_rows.columns:
				vis_order = dup_rows['VISCODE'].astype(str).str.lower().map({'sc': 0, 'init': 1}).fillna(2)
				earliest_row = dup_rows.loc[vis_order.idxmin()]
			else:
				earliest_row = dup_rows.iloc[0]
			# Drop other duplicates
			rows_to_drop = dup_rows.index.difference([earliest_row.name])
			MMSE_df = MMSE_df.drop(rows_to_drop)
	# Check missing subjects
	missing_subs = set(demographics_df['Subject ID'].astype(str).unique()) - set(MMSE_df['PTID'].astype(str).unique())
	if missing_subs:
		print(f"Warning: The following subjects are missing in MMSE data: {missing_subs}")
	
	# Operations
	MMSE_df['MMLAN'] = MMSE_df[['MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR', 'MMREAD', 'MMWRITE', 'MMDRAW']].sum(axis=1, skipna=True)
	MMSE_df['MMORIT'] = MMSE_df[['MMYEAR', 'MMMONTH', 'MMDATE', 'MMDAY', 'MMSEASON']].sum(axis=1, skipna=True)
	MMSE_df['MMORIP'] = MMSE_df[['MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE']].sum(axis=1, skipna=True)
	MMSE_df['MMORI'] = MMSE_df['MMORIT'] + MMSE_df['MMORIP']
	MMSE_df['MMREG'] = MMSE_df[['WORD1', 'WORD2', 'WORD3']].sum(axis=1, skipna=True)
	MMSE_df['MMATTCAL'] = MMSE_df['WORLDSCORE']
	MMSE_df['MMREC'] = MMSE_df[['WORD1DL', 'WORD2DL', 'WORD3DL']].sum(axis=1, skipna=True)
	
	# Get only relevant columns
	MMSE_df = MMSE_df[['PHASE', 'PTID', 'MMLAN', 'MMORIT', 'MMORIP', 'MMORI', 'MMREG', 'MMATTCAL', 'MMREC', 'MMSCORE']]
	# Rename MMSE score columns if present
	MMSE_df = MMSE_df.rename(columns={'MMSCORE': 'MMSE'})

	return MMSE_df
