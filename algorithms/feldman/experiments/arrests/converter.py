import time

def expand_to_one_hot(data,expand = True):
    new_data = []
    for entry in data:
	temp = {}
	if expand == True:
	    if entry["SEX1"] == "FEMALE":
		temp['female'] = 1
	    else:
		temp['female'] = 0

	    if entry["ALCABUS"] == 'INMATE IS AN ALCOHOL ABUSER':
		temp['prior_alcohol_abuse'] = 1
	    else:
		temp['prior_alcohol_abuse'] = 0

	    if entry['DRUGAB'] == 'INMATE IS A DRUG ABUSER':
		temp['prior_drug_abuse'] = 1
	    else:
		temp['prior_drug_abuse'] = 0

	    if entry['NFRCTNS'] == 'INMATE HAS RECORD':
		temp['infraction_in_prison'] = 1
	    else:
		temp['infraction_in_prison'] = 0
    

	    release_age_cats = ['14 TO 17 YEARS OLD','18 TO 24 YEARS OLD', '25 TO 29 YEARS OLD', \
	    '30 TO 34 YEARS OLD','35 TO 39 YEARS OLD','40 TO 44 YEARS OLD','45 YEARS OLD AND OLDER']
	    for cat in release_age_cats:
		if entry['RLAGE'] == cat:
		    temp['release_age_'+cat] = 1
		else:
		    temp['release_age_'+cat] = 0
    
	    time_served_cats = ['None','1 TO 6 MONTHS','13 TO 18 MONTHS','19 TO 24 MONTHS','25 TO 30 MONTHS', \
			'31 TO 36 MONTHS','37 TO 60 MONTHS','61 MONTHS AND HIGHER','7 TO 12 MONTHS']
	    for cat in time_served_cats:
		if entry['TMSRVC'] == cat:
		    temp['time_served_'+cat] = 1
		else:
		    temp['time_served_'+cat] = 0

	    prior_arrest_cats = ['None','1 PRIOR ARREST','11 TO 15 PRIOR ARRESTS','16 TO HI PRIOR ARRESTS','2 PRIOR ARRESTS', \
		'3 PRIOR ARRESTS','4 PRIOR ARRESTS','5 PRIOR ARRESTS','6 PRIOR ARRESTS','7 TO 10 PRIOR ARRESTS']
	    for cat in prior_arrest_cats:
		if entry['PRIRCAT'] == cat:
		    temp['prior_arrest_'+cat] = 1
		else:
		    temp['prior_arrest_'+cat] = 0

	    conditional_release =['PAROLE BOARD DECISION-SERVED NO MINIMUM','MANDATORY PAROLE RELEASE', 'PROBATION RELEASE-SHOCK PROBATION', \
			'OTHER CONDITIONAL RELEASE']
	    unconditional_release = ['EXPIRATION OF SENTENCE','COMMUTATION-PARDON','RELEASE TO CUSTODY, DETAINER, OR WARRANT', \
			'OTHER UNCONDITIONAL RELEASE']
	    other_release = ['NATURAL CAUSES','SUICIDE','HOMICIDE BY ANOTHER INMATE','OTHER HOMICIDE','EXECUTION','OTHER TYPE OF DEATH', \
		    'TRANSFER','RELEASE ON APPEAL OR BOND','OTHER TYPE OF RELEASE','ESCAPE','ACCIDENTAL INJURY TO SELF','UNKNOWN']
	    if entry['RELTYP'] in conditional_release:
		temp['released_conditional'] = 1
		temp['released_unconditional'] = 0
		temp['released_other'] = 0
	    elif entry['RELTYP'] in unconditional_release:
		temp['released_conditional'] = 0
		temp['released_unconditional'] = 1
		temp['released_other'] = 0
	    else:
		temp['released_conditional'] = 0
		temp['released_unconditional'] = 0
		temp['released_other'] = 1

	    try:
		bdate = datetime.date(int(entry['YEAROB2']),int(entry['MNTHOB2']), int(entry['DAYOB2']))
		first_arrest = datetime.date(int(entry['A001YR']),int(entry['A001MO']),int(entry['A001DA']))
		first_arrest_age = first_arrest - bdate
		temp['age_1st_arrest'] = first_arrest_age.days
	    except:
		temp['age_1st_arrest'] = entry['age_1st_arrest']

	    # Add in the Y values
	    temp['Classarrests'] = entry['Classarrests']
	    temp['Classgeneral_violence'] = entry['Classgeneral_violence']
	    temp['Classfatal_violence'] = entry['Classfatal_violence']
	    temp['Classproperty'] = entry['Classproperty']
	    temp['Classsexual_violence'] = entry['Classsexual_violence']
	    temp['Classdrug'] = entry['Classdrug']
	else:
	    temp['SEX1'] = entry['SEX1']
	    temp['RELTYP'] = entry['RELTYP']
	    temp['PRIRCAT'] = entry['PRIRCAT']
	    temp['ALCABUS'] = entry['ALCABUS']
	    temp['DRUGAB'] = entry['DRUGAB']
	    temp['RLAGE'] = entry['RLAGE']
	    temp['TMSRVC'] = entry['TMSRVC']
	    temp['NFRCTNS'] = entry['NFRCTNS']
	    try:
		bdate = datetime.date(int(entry['YEAROB2']),int(entry['MNTHOB2']), int(entry['DAYOB2']))
		first_arrest = datetime.date(int(entry['A001YR']),int(entry['A001MO']),int(entry['A001DA']))
		first_arrest_age = first_arrest - bdate
		temp['age_1st_arrest'] = first_arrest_age.days
	    except:
		temp['age_1st_arrest'] = 0
    
	new_data.append(temp)


    return new_data
