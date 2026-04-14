* Set project specific folder (user-specific / Hard-coded)
cd "E:\Documents\DEJONGO\Project_Daniel_Lewis"
* Set the root directory
global rootdir : pwd

**************************************************************************************************************************************
**************************************************************************************************************************************
* Define a location where we will hold all ado-packages
global adodir "$rootdir/ado"
* Make sure it exists, if not create it
cap mkdir "$adodir"
* Display current system directories (for reference)
sysdir
* Remove OLDPLACE and PERSONAL from the adopath
capture adopath - OLDPLACE
capture adopath - PERSONAL
* Set the PLUS directory to our custom ado directory
sysdir set PLUS "$adodir"
* Add PLUS to the beginning of the adopath
adopath ++ PLUS
* Verify the new adopath setup
adopath
** needed ado-packages: filelist, winsor2, ftools, reghdfe, require, gtools
* "ssc install" these if not yet in "$adodir"
**************************************************************************************************************************************
**************************************************************************************************************************************
clear
*location of the current set of routines, outputfiles
global basepath= "${rootdir}\999_script_to_share"
*list of countries in sample, used in loops
global cntrylst = "AT BE"

* BE ES FR GR IT NL PT DE"
local IndepVars_BFT 	l_Fixed l_Secured l_TermWC  l_BankShareinFirm 
local IndepVars_BRST 	l_BST_presence l_BST_exposure
****************************************************************************************************************
****************************************************************************************************************
*** create dta and output folder, if they do not yet exist
capture shell mkdir "${basepath}\dta"
capture shell mkdir "${basepath}\output"

*** create subfolder in dta, or empty that specific folder if it already exists
local path "${basepath}\dta\DS_shocks"
filelist, dir("`path'") pattern("*.dta")
count if !missing(filename)
if r(N)==0 {
    di "Folder does NOT exist — will create."
    shell mkdir "`path'"
}
else if r(N)>=0{
    di "Folder exists — emptying."
    shell del /q "`path'\*.*"
}
****************************************************************************************************************	
****************************************************************************************************************	
****************************************************************************************************************	
foreach country of global cntrylst {
	forvalues time=1/3{

		if `time'==1 {
			local start_period	= 201909
			local end_period	= 202012
		}
		else if `time'==2 {
			local start_period	= 202103
			local end_period	= 202206
		}
		else if `time'==3 {
			local start_period	= 202209
			local end_period	= 202312
		}
		
		use  if dt_rfrnc>= `start_period' & dt_rfrnc<= `end_period' 				using "${rootdir}\02d_dta_to_csv_for_matlab\\`country'_long_changes0_2019_2023.dta", clear
		merge m:1 dt_rfrnc debt_entty_riad_cd_le 									using "${rootdir}\02b_dta_FT\FT_`country'_2018_2023.dta" , keepusing( debt_nace2d debt_trrtrl_unt_le) keep(1 3) nogen
		merge 1:1 dt_rfrnc debt_entty_riad_cd_le cred_entty_riad_cd_le 				using "${rootdir}\05_sources_variation_dP_dlnQ\dta\temp\FBT_`country'_2018_2023_wlag.dta" , keepusing(`IndepVars_BFT') keep(1 3) nogen
		merge m:1 dt_rfrnc cred_entty_riad_cd_le debt_trrtrl_unt_le debt_nace2d 	using "${rootdir}\05_sources_variation_dP_dlnQ\dta\temp\BRST_`country'_2018_2023_wlag.dta" , keepusing(`IndepVars_BRST') keep(1 3) nogen

		*** pre-process the data, e.g.:
		* - winsorizing
		* - demeaning / partialling out fixed effects   
		quietly foreach v of varlist dQdh dP {
			* winsorize	
			winsor2 `v', cuts(5.0 95.0) by(dt_rfrnc)
			* run a regression and save residuals (can be used for time-demeaning, partialling out covariates, partialling out fixed effects)
			reghdfe `v'_w `IndepVars_BFT'  `IndepVars_BRST', a(dt_rfrnc) residuals(`v'_res)   // <-- as in paper, time-demeaning and partialling out lagged relationship characteristics
			drop `v' `v'_w
			rename `v'_res  `v'
			replace `v'=100*`v'
		}
		drop `IndepVars_BFT' `IndepVars_BRST'  debt_nace2d debt_trrtrl_unt_le
		drop if dQdh==. | dP==.
		
		{
			*** create indices (convenient for Python/Matlab rather than firm/bank identifiers)
			gegen Firm=group(debt_entty_riad_cd_le) 	// in order to have values from 1,..., F (number of firms)
			gegen Bank=group(cred_entty_riad_cd_le)		// in order to have values from 1,..., B (number of banks)
			gegen Time=group(dt_rfrnc)					// in order to have values from 1,..., T (number of time periods)

			** preserve the dataset. Subsequently, a minimalistic dataset is kept to run the python script that computes Ahat and the shocks.
			preserve
			keep 	Firm Bank Time dQdh dP 
			order 	Firm Bank Time dQdh dP 
			// trick to not break the loop if dataset is empty (unavailable for a country or time period)
			quietly sum Bank
			if r(max)>1 {
				export delimited using "${basepath}\csv_FBT_panel_QP_long", replace  // create a csv file with quintuple (F,B,T,dQ,dP) - crucial entry into Python script
				// call Python script via a shell command. Define three environment variables (COUNTRY, START_PERIOD, END_PERIOD) - can be hard-coded if run for single country/time period
				shell cmd /c "set COUNTRY=`country' && set START_PERIOD=`start_period' && set END_PERIOD=`end_period' &&  python "${basepath}/python_Supply_Demand_AKM_restrict_LamFFnorm.py"
				// subsequently load the csv file generated by the Python script, which appends the relationship-specific Supply and Demand shock to the quintuple (F,B,T,dQ,dP)
				import delimited "${basepath}\csv_FBT_panel_QPSD_long.csv", case(preserve) clear
				save "${basepath}\temp.dta", replace
			}
			else if r(max) <=1 {
				gen Supply=.
				gen Demand=.
				clonevar Q=dQdh 
				clonevar P=dP
				drop dQdh dP
				save "${basepath}\temp.dta", replace				
			}
			restore
		}

		** merge in the estimated shocks in the preserved dataset
		quietly {
			merge 1:1 Firm Bank Time   using "${basepath}\temp.dta", gen(merge)
			* some sanity checks on the merged-in data.
			* Are Q and dQdh correctly mapped (don't merge on Q due to rounding differences in Stata/Python, but check correlation)
			corr dQdh  Q
			local check1=r(cov_12)
			* Are P and dP correctly mapped  (don't merge on P due to rounding differences in Stata/Python, but check correlation)
			corr dP P
			local check2=r(cov_12)
			* Is the merge between the input and outputfile (of python script) perfect (only merge==3)
			sum merge
			local check3=r(mean)

			* run three joint quality checks!
			if `check1'>0.99999  & `check2'>0.99999 & `check3'==3 {
				drop Q P merge Firm Bank Time
			}
		}			
		save "${basepath}\dta\DS_shocks\SupDemand_`country'_`start_period'_`end_period'.dta", replace
		
		quietly {
			shell erase "${basepath}\temp.dta"
			shell erase "${basepath}\csv_FBT_panel_QP_long.csv"
			shell erase "${basepath}\csv_FBT_panel_QPSD_long.csv"
		}
	}
}
***************************************************************************************************************	
shell copy		"${basepath}\summary_output.csv" 			"${basepath}\output\summary_output_AKMrestrict_LamFFnorm.csv"
shell erase		"${basepath}\summary_output.csv"
***************************************************************************************************************	
***************************************************************************************************************	
