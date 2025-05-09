rm(list=ls())
setwd("/rds/general/project/hda_24-25/live/TDS/Group03/")

## put all libraries here 
library(data.table)
library(dplyr)
library(ggplot2)
library(ccoptimalmatch)

#-----------------------------
### PREPROCESSING
  
# loading biobank data 

ukb_data=readRDS("Datasets/ukb_0_anxiety_kept_non_processed.rds") ## maybe change this to 0 anxiety .rds
ukb_data_no_anxiety=readRDS("Datasets/ukb_0_anxiety_remove_non_processed.rds") ## Dataset without the anxiety related variables


# to only take 0.0 columns i.e. first time/baseline measurements 
ukb_data <- ukb_data %>%
  select(matches("0\\.0$"))
ukb_data_no_anxiety <- ukb_data_no_anxiety %>%
  select(matches("0\\.0$"))

dim(ukb_data)
dim(ukb_data_no_anxiety)

#Subset ukb only to people with GP prescriptions
gp_prescription_data <- read.delim(paste0("/rds/general/project/hda_24-25/live/TDS/Group03/GP_data_Project1/",
                    "gp_scripts.txt"), header = TRUE, sep = "\t",quote="")
gp_prescription_data$issue_date = dmy(gp_prescription_data$issue_date)# Fix date

#filtered_data <- gp_prescription_data[gp_prescription_data$issue_date > as.Date('2006-01-01'), ]
#gp_prescription_filtered <- subset(gp_prescription_data, eid %in% unique(filtered_data$eid))

#gp_patient_eid <- unique(gp_prescription_filtered$eid)
ukb_data <- ukb_data[rownames(ukb_data) %in% unique(gp_prescription_data$eid), ]


#-------------------------------------------
#RECODING  

#Smoking (current smoker were not questioned on past and household. So by default, yes in the past and yes in household)
ukb_data <- ukb_data %>%
  mutate(
    past_smoke.0.0 = ifelse(current_smok.0.0 == 1, 1, past_smoke.0.0),
    smokers_in_household.0.0 = ifelse(current_smok.0.0 == 1, 1, smokers_in_household.0.0)
  )

ukb_data <- ukb_data %>%
  mutate(
    `Pack years of smoking.0.0` = ifelse(`Ever smoked.0.0` == 0, 0, `Pack years of smoking.0.0`),
  )

#Duplicate of night shift work columns
ukb_data <- ukb_data %>%
  select(-`Night shifts worked.0.0`) 

#Night shifts (only for people in paid employment or self-employed). 
#Transform NA answer of non concerned workers to 0 (i.e no night shifts)
ukb_data <- ukb_data %>%
  mutate(
    night_shift_work.0.0 = ifelse(`Current employment status.0.0` != 1, 0, night_shift_work.0.0),
  )

#Cannabis intake 
#Transform NA answer of non smokers to 0 (i.e never smoked)
ukb_data <- ukb_data %>%
  mutate(
    `Maximum frequency of taking cannabis.0.0` = ifelse(`Ever taken cannabis.0.0` == 0, 0, `Maximum frequency of taking cannabis.0.0`),
  )


ukb_data <- ukb_data %>%
  mutate(
    income_score = coalesce(`Income score (England).0.0`, `Income score (Wales).0.0`, `Income score (Scotland).0.0`),
    employment_score = coalesce(`Employment score (England).0.0`, `Employment score (Wales).0.0`),
    health_score = coalesce(`Health score (England).0.0`, `Health score (Wales).0.0`, `Health score (Scotland).0.0`),
    education_score = coalesce(`Education score (England).0.0`, `Education score (Wales).0.0`, `Education score (Scotland).0.0`),
    housing_score = coalesce(`Housing score (England).0.0`, `Housing score (Wales).0.0`, `Housing score (Scotland).0.0`),
    physical_env_score = coalesce(`Physical environment score (Wales).0.0`),
    community_safety_score = coalesce(`Community safety score (Wales).0.0`),
    IMD_score = coalesce(`Index of Multiple Deprivation (England).0.0`, `Index of Multiple Deprivation (Wales).0.0`, `Index of Multiple Deprivation (Scotland).0.0`)
  ) %>%
  select(-c(
    `Income score (England).0.0`, `Income score (Wales).0.0`, `Income score (Scotland).0.0`,
    `Employment score (England).0.0`, `Employment score (Wales).0.0`,
    `Health score (England).0.0`, `Health score (Wales).0.0`, `Health score (Scotland).0.0`,
    `Education score (England).0.0`, `Education score (Wales).0.0`, `Education score (Scotland).0.0`,
    `Housing score (England).0.0`, `Housing score (Wales).0.0`, `Housing score (Scotland).0.0`,
    `Physical environment score (Wales).0.0`,
    `Community safety score (Wales).0.0`,
    `Index of Multiple Deprivation (England).0.0`, `Index of Multiple Deprivation (Wales).0.0`, `Index of Multiple Deprivation (Scotland).0.0`
  ))


ukb_data <- rename(ukb_data, "Crime score.0.0"= "Crime score (England).0.0")
ukb_data <- rename(ukb_data, "Living environment score.0.0"= "Living environment score (England).0.0")
# Check if the new columns exist and the old ones are removed
colnames(ukb_data)


#-----------------------------------------------------
#MISSINGESS
  
  
#missingness for each var, remove individuals at defined missingness threshold of 50% 
#except for variables where it makes sense to have <50% e.g. menopause, certain diagnoses, dependent questions, look into tomorrow 07/02

anxiety_variables<- colnames(ukb_data)[!colnames(ukb_data) %in% colnames(ukb_data_no_anxiety)] 

column_missing <- colMeans(is.na(ukb_data)) * 100
column_missing <- as.data.frame(column_missing)
column_missing$names <- rownames(column_missing)


#remove missing vars unless in anxiety (sex aswell but missingness less than 50)
keep_vars <- rownames(column_missing)[column_missing[,1] <= 50 ]
remove_vars <- rownames(column_missing)[column_missing[,1] > 50 ]
bio_non_remove <- c(remove_vars[129:200],remove_vars[202:204])

ukb_data <- ukb_data[,c(keep_vars,bio_non_remove) ]
dim(ukb_data)

saveRDS(ukb_data, "Datasets/bio_PROCESSED_ukb_data.rds")

list_biomarkers_var <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Variables_blocks/list_biomarkers_var.rds")

saveRDS(c(list_biomarkers_var$list_biomarkers,bio_non_remove), "list_bio_var_global.rds")

global_list_bio <- readRDS("list_bio_var_global.rds")

remove_bio_var=c("Antigen assay date.0.0","Plate used for sample run.0.0", "T. gondii seropositivity for Toxoplasma gondii.0.0", "C. trachomatis Definition I seropositivity for Chlamydia trachomatis.0.0", 
                 "C. trachomatis Definition II seropositivity for Chlamydia trachomatis.0.0", "H. pylori Definition I seropositivity for Helicobacter pylori.0.0", "H. pylori Definition I seropositivity for Helicobacter pylori.0.0",
                 "H. pylori Definition II seropositivity for Helicobacter pylori.0.0")

global_list_bio<-global_list_bio[-which(global_list_bio %in% remove_bio_var)]

saveRDS(global_list_bio, "list_bio_var_global.rds")

#--------------------------------------------
ukb_data <- readRDS("Datasets/bio_PROCESSED_ukb_data.rds")

ukb_data<-ukb_data[, !colnames(ukb_data) %in% remove_bio_var]

#RECODING CAT
ukb_data <- ukb_data %>%
  mutate(
    Transgender = ifelse(sex.0.0 != `Genetic sex.0.0`, 1, 0),
  )

#Neurop because no Field ID
ukb_data <- ukb_data[,!colnames(ukb_data) %in% c("sex.0.0","GP registration records.0.0", "GP prescription records.0.0","GP clinical event records.0.0", "Neuroticism score.0.0")]
categorical_vars <- names(Filter(is.integer, ukb_data))
category_counts <- sapply(ukb_data[categorical_vars], function(x) length(unique(x)))
filtered_vars <- setdiff(names(category_counts[category_counts < 10]), 
                         c("time_watch television.0.0", "time_watch computer.0.0", 
                           "weekly_usage_phone.0.0", "number_of_children.0.0", 
                           "Traffic intensity on the nearest road.0.0", "Close to major road.0.0"))
filtered_vars <- c(filtered_vars,"Illnesses of father.0.0","Illnesses of mother.0.0", 
                   "friend_family visits.0.0","past_smoke.0.0","vitamins_supplements.0.0", 
                   "Illnesses of siblings.0.0","Current employment status.0.0", "weekly_usage_phone.0.0", 
                   "moderate_phys_act.0.0","vigorous_phys_act.0.0", "non_cancer_illness.0.0", "Ethnic background.0.0",
                   "Home area population density - urban or rural.0.0","Close to major road.0.0")
ukb_data[, filtered_vars] <- lapply(ukb_data[, filtered_vars], as.factor)


current_year <- as.numeric(format(Sys.Date(), "%Y"))
ukb_data$Age <- round(current_year - ukb_data$YOR.0.0)
ukb_data <- ukb_data[, !names(ukb_data) %in% "YOR.0.0"]

#HERE
recoded_ukb_data <- ukb_data

choices<-readRDS("extraction_and_recoding/outputs/annot.rds")

#modified_col <- c("income_score","employment_score","health_score", "education_score","housing_score","physical_env_score", "community_safety_score", "IMD_score", "Transgender")
#recoded_data <- recoded_ukb_data[, !(colnames(recoded_ukb_data) %in% modified_col)]

recoded_data <- recoded_ukb_data[,filtered_vars]

for (k in 1:ncol(recoded_data)) {
  tmp_coding_name <- gsub("\\..*", "", colnames(recoded_data)[k])
  print(tmp_coding_name)
  tmp_coding_id <- choices[tmp_coding_name, "Coding"]
  
  coding <- read.table(paste0("/rds/general/project/hda_24-25/live/TDS/Group03/extraction_and_recoding/parameters/codings/codes_", tmp_coding_id, ".txt"), header = TRUE, stringsAsFactors = FALSE)
  coding$RecodedValue[coding$RecodedValue == "NA"] <- NA
  coding$RecodedMeaning[coding$RecodedMeaning == "NA"] <- NA
    
  recoding <- coding$RecodedMeaning
  names(recoding) <- coding$OriginalValue
  recoded_data[, k] <- recoding[as.character(recoded_data[, k])]
    
  if (grepl("categorical", tolower(choices[tmp_coding_name, "ValueType"]))) {
      # Recoding for categorical variables: levels are ordered as indicated by RecodedValue
      recoded_data[, k] <- factor(recoded_data[, k], levels = unique(coding$RecodedMeaning[sort.list(as.numeric(coding$RecodedValue))]))
    }
}
recoded_ukb_data[,filtered_vars] <- recoded_data
recoded_ukb_data$Transgender <- factor(recoded_ukb_data$Transgender, level=c(0,1), label=c("Non-trans", "Trans"))

recoded_ukb_data$moderate_phys_act.0.0 <- as.factor(recoded_ukb_data$moderate_phys_act.0.0)
recoded_ukb_data$vigorous_phys_act.0.0 <- as.factor(recoded_ukb_data$vigorous_phys_act.0.0)

recoded_ukb_data <- recoded_ukb_data %>% 
  mutate(across(everything(), ~ if (is.factor(.)) . else ifelse(. == -10, 0.5, .)))
recoded_ukb_data <- recoded_ukb_data %>% 
  mutate(across(everything(), ~ if (is.factor(.)) . else ifelse(. == -1, NA, .)))
recoded_ukb_data <- recoded_ukb_data %>% 
  mutate(across(everything(), ~ if (is.factor(.)) . else ifelse(. == -3, NA, .)))

saveRDS(recoded_ukb_data, "Datasets/Recoded_ukb_data_bio.rds")

#-------------------------------------------
#SPLITTING
  
ukb_data <- readRDS("Datasets/Recoded_ukb_data_bio.rds")
  
# Split dataset by sex
ukb_data_male <- subset(ukb_data, `Genetic sex.0.0` == "Male")
ukb_data_female <- subset(ukb_data, `Genetic sex.0.0` == "Female")
sex_variables <- c("menopause.0.0", "number_of_children.0.0", "contraceptive_pill.0.0")

# Remove the 3 sex-dependent columns from the male dataset
ukb_data_male <- ukb_data_male[, !(colnames(ukb_data_male) %in% sex_variables)]

saveRDS(ukb_data_male, "Datasets/PROCESSED_ukb_data_male_bio.rds")
saveRDS(ukb_data_female, "Datasets/PROCESSED_ukb_data_female_bio.rds")


#---------------------------------------------------------------------------
  # DECIDE AND SEPERATE CASES-CONTROLS -----------------------------------------------------

GP_data <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/GP_data_Project1/GP_prescriptions_benzos_above2006.rds")
date <- as.Date(GP_data$issue_date)

library(dplyr)
library(lubridate)

PROCESSED_ukb_data_female <- readRDS("Datasets/PROCESSED_ukb_data_female_bio.rds")
PROCESSED_ukb_data_male <- readRDS("Datasets/PROCESSED_ukb_data_male_bio.rds")


# Transform issue_date into date format
data <- GP_data %>%
  mutate(issue_date = as.Date(issue_date))

# Calculate the time interval of prescription, group by each patient
# date interval <= 365 defined as case, others as controls, save as a list
case_control_list <- data %>%
  arrange(eid, issue_date) %>%
  group_by(eid) %>%
  mutate(date_diff = issue_date - lag(issue_date)) %>%
  summarise(case_control = ifelse(any(date_diff <= 365, na.rm = TRUE), "case", "control")) %>%
  ungroup()

# View the list
head(case_control_list)

# save to file
getwd()  
setwd("/rds/general/project/hda_24-25/live/TDS/Group03/GP_data_Project1")  
saveRDS(case_control_list, "case_control_list_bio.rds")

# Change directory to "Dataset"
setwd("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets")

## 1 Female
# make sure eid as a column
ukb_data_female <- PROCESSED_ukb_data_female %>%
  mutate(eid = as.character(rownames(PROCESSED_ukb_data_female))) 

# make sure eid of case_control_list also save as character
case_control_list <- case_control_list %>%
  mutate(eid = as.character(eid))

#  left_join matching
ukb_matched_female <- ukb_data_female %>%
  left_join(case_control_list, by = "eid") %>%
  mutate(case_control = ifelse(is.na(case_control), "control", case_control))  

ukb_matched_female <- ukb_matched_female %>%
  mutate(case_control = ifelse(case_control == "case", 1, 0))  # case → 1, control → 0


## 2 Male
# make sure eid as a column
ukb_data_male <- PROCESSED_ukb_data_male %>%
  mutate(eid = as.character(rownames(PROCESSED_ukb_data_male)))  

# left_join matching
ukb_matched_male <- ukb_data_male %>%
  left_join(case_control_list, by = "eid") %>%
  mutate(case_control = ifelse(is.na(case_control), "control", case_control))  

ukb_matched_male <- ukb_matched_male %>%
  mutate(case_control = ifelse(case_control == "case", 1, 0))  # case → 1, control → 0


## 3 Overall #### DONT NEED THIS OVERALL ONE 
# make sure eid as a column
#ukb_data_anxiety <- PROCESSED_ukb_data_anxiety %>%
  #mutate(eid = as.character(rownames(PROCESSED_ukb_data_anxiety)))  

# left_join matching
#ukb_matched_anxiety <- ukb_data_anxiety %>%
  #left_join(case_control_list, by = "eid") %>%
 # mutate(case_control = ifelse(is.na(case_control), "control", case_control)) 

#ukb_matched_anxiety <- ukb_matched_anxiety %>%
  #mutate(case_control = ifelse(case_control == "case", 1, 0))  # case → 1, control → 0

table(case_control_list$case_control)
table(ukb_matched_female$case_control)
table(ukb_matched_male$case_control)
#table(ukb_matched_anxiety$case_control)
#saveRDS(ukb_matched_anxiety, "ukb_matched_anxiety.rds")
saveRDS(ukb_matched_female, "ukb_matched_female_bio.rds")
saveRDS(ukb_matched_male, "ukb_matched_male_bio.rds")

#---------Filtering controls repeated BZD prescriptions before 2006 ------
ukb_matched_female <- readRDS("ukb_matched_female_bio.rds")
ukb_matched_male <- readRDS("ukb_matched_male_bio.rds")
gp_prescription_data <- read.delim(paste0("/rds/general/project/hda_24-25/live/TDS/Group03/GP_data_Project1/",
                                          "gp_scripts.txt"), header = TRUE, sep = "\t",quote="")
gp_prescription_data$issue_date = dmy(gp_prescription_data$issue_date)# Fix date

benzo_drugs_names<-c("loprazolam","lorazepam","diazepam","clonazepam","alphrazolam","temazepam","Chlordiazepoxide","midazolam","flurazepam","oxazepam","triazolam","clorazepate","clobazam","estazolam","quazepam","remimazolam")
benzo_pattern <- paste(benzo_drugs_names, collapse = "|")
gp_prescription_benzos=gp_prescription_data[grepl(benzo_pattern,x=gp_prescription_data$drug_name,ignore.case = T),]

filtered_data_benzos <- gp_prescription_benzos[gp_prescription_benzos$issue_date < as.Date('2006-01-01'), ]
gp_prescription_benzos_filtered <- subset(gp_prescription_benzos, eid %in% unique(filtered_data_benzos$eid))

exclude_control_list <- gp_prescription_benzos_filtered %>%
  arrange(eid, issue_date) %>%
  group_by(eid) %>%
  mutate(date_diff = issue_date - lag(issue_date)) %>%
  summarise(case_control = ifelse(any(date_diff <= 365, na.rm = TRUE), "case", "control")) %>%
  ungroup()

table(exclude_control_list$case_control)
eid_remove <- exclude_control_list[exclude_control_list$case_control == "case",]

new_df_female<-ukb_matched_female[!ukb_matched_female$eid %in% eid_remove$eid,]
new_df_male<-ukb_matched_male[!ukb_matched_male$eid %in% eid_remove$eid,]

saveRDS(new_df_female, "ukb_matched_female_filtered_bio.rds")
saveRDS(new_df_male, "ukb_matched_male_filtered_bio.rds")
---------------------------------------------------------------------------
  # MATCHING CASES-CONTROLS -----------------------------------------------------

#FILTERING FOR LESS THAN 50% MISSINGNESS FOR BIOMARKERS
new_df_female<-readRDS("Datasets/ukb_matched_female_filtered_bio.rds")
new_df_male<-readRDS("Datasets/ukb_matched_male_filtered_bio.rds")

global_list_bio <- readRDS("../list_bio_var_global.rds")

non_missing_bio_male<-rownames(new_df_male)[rowSums(is.na(new_df_male[,global_list_bio]))/length(global_list_bio)<0.5]
non_missing_bio_female<-rownames(new_df_female)[rowSums(is.na(new_df_female[,global_list_bio]))/length(global_list_bio)<0.5]

new_df_female<- new_df_female[non_missing_bio_female,]
new_df_male<- new_df_male[non_missing_bio_male,]

column_missing_male <- colMeans(is.na(new_df_male)) * 100
column_missing_male <- as.data.frame(column_missing_male)
column_missing_male$names <- rownames(column_missing_male)
keep_vars_male <- rownames(column_missing_male)[column_missing_male[,1] <= 50 ]

new_df_male <- new_df_male[,keep_vars_male ]

column_missing_female <- colMeans(is.na(new_df_female)) * 100
column_missing_female <- as.data.frame(column_missing_female)
column_missing_female$names <- rownames(column_missing_female)
keep_vars_female <- rownames(column_missing_female)[column_missing_female[,1] <= 50 ]

new_df_female <- new_df_female[,keep_vars_female ]


saveRDS(new_df_female, "ukb_matched_female_filtered_bio_miss.rds")
saveRDS(new_df_male, "ukb_matched_male_filtered_bio_miss.rds")

#FEMALE DATA --------------------------------------------------
ukb_female <- readRDS("/rds/general/user/tt1024/projects/hda_24-25/live/TDS/Group03/Datasets/ukb_matched_female_filtered_bio_miss.rds")

# Step 1: Prepare Cases and Controls

# Convert case-control column to factor for proper matching
ukb_female$case_control <- as.factor(ukb_female$case_control)

# Define cases and controls separately
bdd_cases <- ukb_female %>% filter(case_control == "1")   # Select cases
bdd_controls <- ukb_female %>% filter(case_control == "0") # Select controls

# Assign unique case IDs
bdd_cases$cluster_case <- paste("case", seq_len(nrow(bdd_cases)), sep = "_")

# Controls should not have a cluster_case value initially (they will be assigned later)
bdd_controls$cluster_case <- NA

# Convert to data.table for speed
setDT(bdd_cases)
setDT(bdd_controls)

# Rename age column in cases and controls
setnames(bdd_cases, "Age", "case_age")
setnames(bdd_controls, "Age", "control_age")

# Duplicate the control age column so you can keep it after merging
bdd_controls[, control_age_copy := control_age]

# Rename the original control age column for use as the join key.
setnames(bdd_controls, "control_age", "control_age_original")

# Perform merge using the renamed join key
matched_pairs <- merge(
  bdd_cases[, .(cluster_case, eid, case_control, case_age)],  # Cases
  bdd_controls[, .(eid, case_control, control_age_original, control_age_copy)],  # Controls with both age columns
  by.x = "case_age", by.y = "control_age_original",  # Merge on age
  allow.cartesian = TRUE
)

# Rename columns for clarity
setnames(matched_pairs, 
         c("eid.x", "eid.y", "case_control.x", "case_control.y"), 
         c("case_eid", "control_eid", "case_control", "control_case_control"))

# Compute age difference correctly
matched_pairs[, age_diff := abs(case_age - control_age_copy)]
summary(matched_pairs$age_diff)  # Check min, max, mean, etc.

# Count the number of controls matched for each case
controls_per_case <- matched_pairs[, .N, by = cluster_case]

# Check if every case has at least 3 controls
if (all(controls_per_case$N >= 3)) {
  print("Every case has at least 3 controls.")
} else {
  print("Not every case has at least 3 controls. The following cases have fewer than 3 controls:")
  print(controls_per_case[N < 3])
}

# For each case (grouped by cluster_case), sample 3 controls without replacement
sampled_matches <- matched_pairs[, .SD[sample(.N, 3)], by = cluster_case]

# Verify that each case now has exactly 3 controls
controls_per_case_after <- sampled_matches[, .N, by = cluster_case]
print(controls_per_case_after)

#SUBSETTING UKB_FEMALE SO ONLY MATCHED ONES REMAIN
setDT(ukb_female)
# 1. Identify all eids present in sampled_matches.
selected_eids <- unique(c(sampled_matches$case_eid, sampled_matches$control_eid))
# 2. Subset ukb_female to only keep rows with these eids.
print(colnames(ukb_female))
ukb_matched <- ukb_female[eid %in% selected_eids]
# 3. Create a lookup table that maps each eid to its cluster_case.
# For cases:
case_lookup <- unique(sampled_matches[, .(eid = case_eid, cluster_case)])
# For controls:
control_lookup <- unique(sampled_matches[, .(eid = control_eid, cluster_case)])
# Combine the two (in case a case's eid and a control's eid both appear):
lookup <- unique(rbind(case_lookup, control_lookup))
# 4. Merge the lookup table into the subsetted data to add the cluster_case column.
ukb_matched_female <- merge(ukb_matched, lookup, by = "eid", all.x = FALSE)

saveRDS(ukb_matched_female, "NEW_ukb_match_fem_final_bio.rds")

#MALE DATA (variable names are the same as female) --------------------------

ukb_male <- readRDS("/rds/general/user/tt1024/projects/hda_24-25/live/TDS/Group03/Datasets/ukb_matched_male_filtered_bio_miss.rds")
# Step 1: Prepare Cases and Controls

# Convert case-control column to factor for proper matching
ukb_male$case_control <- as.factor(ukb_male$case_control)

# Define cases and controls separately
bdd_cases <- ukb_male %>% filter(case_control == "1")   # Select cases
bdd_controls <- ukb_male %>% filter(case_control == "0") # Select controls


# Assign unique case IDs
bdd_cases$cluster_case <- paste("case", seq_len(nrow(bdd_cases)), sep = "_")

# Controls should not have a cluster_case value initially (they will be assigned later)
bdd_controls$cluster_case <- NA

# Convert to data.table for speed
setDT(bdd_cases)
setDT(bdd_controls)

# Rename age column in cases and controls
setnames(bdd_cases, "Age", "case_age")
setnames(bdd_controls, "Age", "control_age")

# Duplicate the control age column so you can keep it after merging
bdd_controls[, control_age_copy := control_age]

# Rename the original control age column for use as the join key.
setnames(bdd_controls, "control_age", "control_age_original")

# Perform merge using the renamed join key
matched_pairs <- merge(
  bdd_cases[, .(cluster_case, eid, case_control, case_age)],  # Cases
  bdd_controls[, .(eid, case_control, control_age_original, control_age_copy)],  # Controls with both age columns
  by.x = "case_age", by.y = "control_age_original",  # Merge on age
  allow.cartesian = TRUE
)

print(colnames(matched_pairs))

# Rename columns for clarity
setnames(matched_pairs, 
         c("eid.x", "eid.y", "case_control.x", "case_control.y"), 
         c("case_eid", "control_eid", "case_control", "control_case_control"))

# Compute age difference correctly
matched_pairs[, age_diff := abs(case_age - control_age_copy)]
summary(matched_pairs$age_diff)  # Check min, max, mean, etc.

# Check if the fix worked
head(matched_pairs)

# Count the number of controls matched for each case
controls_per_case <- matched_pairs[, .N, by = cluster_case]

# Look at the counts for each case
print(controls_per_case)
summary(controls_per_case)

# Check if every case has at least 3 controls
if (all(controls_per_case$N >= 3)) {
  print("Every case has at least 3 controls.")
} else {
  print("Not every case has at least 3 controls. The following cases have fewer than 3 controls:")
  print(controls_per_case[N < 3])
}

# For each case (grouped by cluster_case), sample 3 controls without replacement
sampled_matches <- matched_pairs[, .SD[sample(.N, 3)], by = cluster_case]

# Verify that each case now has exactly 3 controls
controls_per_case_after <- sampled_matches[, .N, by = cluster_case]
print(controls_per_case_after)

print(colnames(sampled_matches))
View(sampled_matches)


#SUBSETTING UKB_FEMALE SO ONLY MATCHED ONES REMAIN
setDT(ukb_male)
# 1. Identify all eids present in sampled_matches.
selected_eids <- unique(c(sampled_matches$case_eid, sampled_matches$control_eid))
# 2. Subset ukb_female to only keep rows with these eids.
print(colnames(ukb_male))
ukb_matched <- ukb_male[eid %in% selected_eids]
# 3. Create a lookup table that maps each eid to its cluster_case.
# For cases:
case_lookup <- unique(sampled_matches[, .(eid = case_eid, cluster_case)])
# For controls:
control_lookup <- unique(sampled_matches[, .(eid = control_eid, cluster_case)])
# Combine the two (in case a case's eid and a control's eid both appear):
lookup <- unique(rbind(case_lookup, control_lookup))
# 4. Merge the lookup table into the subsetted data to add the cluster_case column.
ukb_matched_male <- merge(ukb_matched, lookup, by = "eid", all.x = FALSE)
print(colnames(ukb_matched_male))

saveRDS(ukb_matched_male, "NEW_ukb_match_male_final_bio.rds")

#####RECODING######

#load datasets 

project_path=dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(project_path)

ukb_fem <- readRDS("Datasets/NEW_ukb_match_fem_final_bio.rds")
ukb_fem <- as.data.frame(ukb_fem)
ukb_male <- readRDS("Datasets/NEW_ukb_match_male_final_bio.rds")
ukb_male <- as.data.frame(ukb_male)

ukb_fem[sapply(ukb_fem, is.character)] <- 
  lapply(ukb_fem[sapply(ukb_fem, is.character)], as.factor)

ukb_fem[sapply(ukb_fem, is.integer)] <- 
  lapply(ukb_fem[sapply(ukb_fem, is.character)], as.numeric)

ukb_male[sapply(ukb_male, is.character)] <- 
  lapply(ukb_male[sapply(ukb_male, is.character)], as.factor)

ukb_male[sapply(ukb_male, is.integer)] <- 
  lapply(ukb_male[sapply(ukb_male, is.character)], as.numeric)

#### FEMALE #### test train split based on matching 

set.seed(5225678) 

unique_clusters <- unique(ukb_fem$cluster_case)

train_clusters <- sample(unique_clusters, size = 0.8 * length(unique_clusters))

train_fem <- which(ukb_fem$cluster_case %in% train_clusters)
test_fem  <- which(!ukb_fem$cluster_case %in% train_clusters)

Y <- as.factor(ukb_fem$case_control)
X <- as.data.frame(ukb_fem[, !colnames(ukb_fem) %in% c("case_control")])

X_train_fem <- X[train_fem, ]
Y_train_fem <- Y[train_fem]

X_test_fem <- X[test_fem, ]
Y_test_fem <- Y[test_fem]

# Check matching cases kept together
table(ukb_fem$cluster_case[train_fem] %in% ukb_fem$cluster_case[test_fem])  

saveRDS(Y_train_fem, "Datasets/Y_train_fem_bio.rds")
saveRDS(Y_test_fem, "Datasets/Y_test_fem_bio.rds")

## FOR MALES ##

set.seed(5225678) 

unique_clusters <- unique(ukb_male$cluster_case)

train_clusters <- sample(unique_clusters, size = 0.8 * length(unique_clusters))

train_male <- which(ukb_male$cluster_case %in% train_clusters)
test_male  <- which(!ukb_male$cluster_case %in% train_clusters)

Y <- as.factor(ukb_male$case_control)
X <- as.data.frame(ukb_male[, !colnames(ukb_male) %in% c("case_control")])

X_train_male <- X[train_male, ]
Y_train_male <- Y[train_male]

X_test_male <- X[test_male, ]
Y_test_male <- Y[test_male]

# Check matching cases kept together
table(ukb_male$cluster_case[train_male] %in% ukb_male$cluster_case[test_male])  

saveRDS(Y_train_male, "Datasets/Y_train_male_bio.rds")
saveRDS(Y_test_male, "Datasets/Y_test_male_bio.rds")

############# IMPUTATION ##################


remove_vars <- c("eid","Genetic sex.0.0", "cluster_case", "Well used for sample run.0.0", "non_cancer_illness.0.0")
X_toy_train_fem <- X_train_fem[1:100,!colnames(X_train_fem) %in% remove_vars]

#impute training data and extract
imputed_train_male <- missForest(X_toy_train_fem, save_models = TRUE)


remove_vars <- c("eid","Genetic sex.0.0", "cluster_case", "Well used for sample run.0.0", "non_cancer_illness.0.0")
X_train_fem <- X_train_fem[,!colnames(X_train_fem) %in% remove_vars]
X_test_fem <- X_test_fem[,!colnames(X_test_fem) %in% remove_vars]
X_train_male <- X_train_male[,!colnames(X_train_male) %in% remove_vars]
X_test_male <- X_test_male[,!colnames(X_test_male) %in% remove_vars]

#impute training data and extract
imputed_train_male <- missForest(X_train_male, save_models = TRUE)
imputed_train_fem  <- missForest(X_train_fem, save_models = TRUE)

saveRDS(imputed_train_male$ximp, "Datasets/imputed_train_male_X.rds")
saveRDS(imputed_train_fem$ximp, "Datasets/imputed_train_fem_X.rds")

#impute test data

imputed_test_male <- missForestPredict(imputed_train_male, newdata = X_test_male)
imputed_test_fem <- missForestPredict(imputed_train_fem, newdata = X_test_fem)

saveRDS(imputed_test_male, "Datasets/imputed_test_male_X.rds")
saveRDS(imputed_test_fem, "Datasets/imputed_test_fem_X.rds")

############ ONE HOT ENCODING and SCALING ###################
##ONLY KEEP BIOMARKERS

global_list_bio <- readRDS("list_bio_var_global.rds")

#load data

X_train_fem <- readRDS("Datasets/imputed_train_fem_bio_X.rds")
X_test_fem <- readRDS("Datasets/imputed_test_fem_bio_X.rds")

X_train_male <- readRDS("Datasets/imputed_train_male_bio_X.rds")
X_test_male <- readRDS("Datasets/imputed_test_male_bio_X.rds")

X_train_fem <- X_train_fem[,colnames(X_train_fem) %in% global_list_bio]
X_test_fem <- X_test_fem[,colnames(X_test_fem) %in% global_list_bio]

X_train_male <- X_train_male[,colnames(X_train_male) %in% global_list_bio]
X_test_male <- X_test_male[,colnames(X_test_male) %in% global_list_bio]

saveRDS(X_train_fem, "Datasets/X_train_fem_bio.rds")
saveRDS(X_test_fem, "Datasets/X_test_fem_bio.rds")

saveRDS(X_train_male, "Datasets/X_train_male_bio.rds")
saveRDS(X_test_male, "Datasets/X_test_male_bio.rds")

## STANDARDISE 

# male standardise 
train_numeric_vars <- X_train_male %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)
names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_male[, col] <- (X_train_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_male[, col]  <- (X_test_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}


#female standardising
train_numeric_vars <- X_train_fem %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)

names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_fem[, col] <- (X_train_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_fem[, col]  <- (X_test_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}

saveRDS(X_train_fem, "Datasets/X_train_fem_scaled_bio.rds")
saveRDS(X_test_fem, "Datasets/X_test_fem_scaled_bio.rds")

saveRDS(X_train_male, "Datasets/X_train_male_scaled_bio.rds")
saveRDS(X_test_male, "Datasets/X_test_male_scaled_bio.rds")


###LASSO, RANDOM FOREST, ELASTIC NET######









 