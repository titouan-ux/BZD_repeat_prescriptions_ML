library(ggplot2)
library(ComplexUpset)
library(jsonlite)
library(data.table)
library(dplyr)

#CREATION OF THE MATRIX
X_train_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_male_nobio_RF.rds")
features_male <- colnames(X_train_male)
X_train_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_fem_nobio_RF.rds")
features_female <- colnames(X_train_female)

models <- c("Lasso Male","ElNet Male", "RF Male","XGb Male","Lasso Female", "ElNet Female", "RF Female", "XGb Female")

# Create binary presence/absence matrix
binary_matrix <- data.frame(matrix(0, nrow=length(models), ncol=length(features_female)))
colnames(binary_matrix) <- features_female
rownames(binary_matrix) <- models



#top_n <- 20
path_RF <- "/rds/general/project/hda_24-25/live/TDS/Group03/RF"
path_XG <- "/rds/general/project/hda_24-25/live/TDS/Group03/XGboost"

# Random Forest Male
importances_RF_male <- fromJSON(file.path(path_RF, "importance_vars_permutation_male.json"))
#indices_RF_male <- order(importances_RF_male, decreasing = TRUE)[1:top_n]
indices_RF_male<-which(importances_RF_male>0)
#top_features_RF_male <- features_male[indices_RF_male]
top_features_RF_male <- names(indices_RF_male)
binary_matrix["RF Male", top_features_RF_male] <- 1

# Random Forest Female
importances_RF_female <- fromJSON(file.path(path_RF, "importance_vars_permutation_female.json"))
#indices_RF_female <- order(importances_RF_female, decreasing = TRUE)[1:top_n]
indices_RF_female<-which(importances_RF_female>0)
top_features_RF_female <- names(indices_RF_female)
#top_features_RF_female <- features_female[indices_RF_female]
binary_matrix["RF Female", top_features_RF_female] <- 1

# XGBoost Male
q3=11.276260137557983 #3rd quartile value of gain importance
importances_XG_male <- fromJSON(file.path(path_XG, "variable_importance_male.json"))
importances_XG_male <- importances_XG_male[which(importances_XG_male>q3)]
sorted_features_XG <- sort(unlist(importances_XG_male), decreasing = TRUE)
top_features_XG_male <- names(sorted_features_XG)
binary_matrix["XGb Male", top_features_XG_male] <- 1

# XGBoost Female
q3=13.066758155822754 #3rd quartile value of gain importance
importances_XG_female <- fromJSON(file.path(path_XG, "variable_importance_female.json"))
importances_XG_female <- importances_XG_female[which(importances_XG_female>q3)]
sorted_features_XG <- sort(unlist(importances_XG_female), decreasing = TRUE)
top_features_XG_female <- names(sorted_features_XG)
binary_matrix["XGb Female", top_features_XG_female] <- 1

# LASSO Male
variables_selection_LASSO_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stable_vars_lasso_male.rds")
variables_selection_LASSO_male_names <- names(variables_selection_LASSO_male)
# Filter out variables containing "0.0"
non_0_var <- variables_selection_LASSO_male_names[!grepl("0.0", variables_selection_LASSO_male_names)]
print(non_0_var)
variables_selection_LASSO_male_names <- unique(sub("0.0.*", "0.0", variables_selection_LASSO_male_names[!(variables_selection_LASSO_male_names %in% non_0_var)]))
# Add specific variables
variables_selection_LASSO_male_names <- c(variables_selection_LASSO_male_names, "health_score", "Overall_health_rating")
binary_matrix["Lasso Male", variables_selection_LASSO_male_names] <- 1



# LASSO Female
variables_selection_LASSO_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stable_vars_lasso_fem.rds")
variables_selection_LASSO_female_names <- names(variables_selection_LASSO_female)
non_0_var <- variables_selection_LASSO_female_names[!grepl("0.0", variables_selection_LASSO_female_names)]
print(non_0_var)
variables_selection_LASSO_female_names <- unique(sub("0.0.*", "0.0", variables_selection_LASSO_female_names[!(variables_selection_LASSO_female_names %in% non_0_var)]))
variables_selection_LASSO_female_names <- c(variables_selection_LASSO_female_names, "health_score", "Overall_health_rating", "income_score")
binary_matrix["Lasso Female", variables_selection_LASSO_female_names] <- 1


# Elastic Net Female
variables_selection_ELNET_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stable_vars_elnet_female.rds")
variables_selection_ELNET_female_names <- names(variables_selection_ELNET_female)
non_0_var <- variables_selection_ELNET_female_names[!grepl("0.0", variables_selection_ELNET_female_names)]
print(non_0_var)
variables_selection_ELNET_female_names <- unique(sub("0.0.*", "0.0", variables_selection_ELNET_female_names[!(variables_selection_ELNET_female_names %in% non_0_var)]))
variables_selection_ELNET_female_names <- c(variables_selection_ELNET_female_names, "health_score", "Overall_health_rating","income_score")
binary_matrix["ElNet Female", variables_selection_ELNET_female_names] <- 1

# Elastic Net Male
variables_selection_ELNET_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stable_vars_elnet_male.rds")
variables_selection_ELNET_male_names <- names(variables_selection_ELNET_male)
non_0_var <- variables_selection_ELNET_male_names[!grepl("0.0", variables_selection_ELNET_male_names)]
print(non_0_var)
variables_selection_ELNET_male_names <- unique(sub("0.0.*", "0.0", variables_selection_ELNET_male_names[!(variables_selection_ELNET_male_names %in% non_0_var)]))
variables_selection_ELNET_male_names <- c(variables_selection_ELNET_male_names, "health_score", "Overall_health_rating")
binary_matrix["ElNet Male", variables_selection_ELNET_male_names] <- 1

# Drop columns with only zeros
binary_matrix <- binary_matrix[, colSums(binary_matrix) > 0]
binary_matrix[is.na(binary_matrix)] <- 0


saveRDS(binary_matrix,"/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female.rds")

#CREATION OF THE PLOT
#bin_matrix <- read_csv("/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female.csv")
bin_matrix <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female.rds")
sex_indicator <- c("Female", "Male", "Female", "Male", "Female","Male", "Female","Male")
# bin_matrix$sex <- ifelse(sex_indicator=="Male",1,0)

bin_matrix_df <- as.data.frame(bin_matrix)
bin_matrix_df<-bin_matrix_df[,colnames(bin_matrix_df)[colSums(bin_matrix_df)!=1]]
colnames(bin_matrix_df) <- gsub("\\.0\\.0$", "", colnames(bin_matrix_df))  # Remove "0.0" at the end
colnames(bin_matrix_df) <- gsub("_", " ", colnames(bin_matrix_df))               # Replace underscores with spaces

# Capitalize the first letter of each word
colnames(bin_matrix_df) <- sapply(strsplit(colnames(bin_matrix_df), " "), function(x) {
  paste(toupper(substring(x, 1, 1)), substring(x, 2), sep = "", collapse = " ")
})

# Convert to a vector
colnames(bin_matrix_df) <- as.vector(colnames(bin_matrix_df))
bin_matrix_df<-rename(bin_matrix_df,"Seen GP for Nerves/Anxiety/Tension/Depression"="Seen Doctor (GP) For Nerves, Anxiety, Tension Or Depression")
bin_matrix_df<-rename(bin_matrix_df,"Seen Psychiatrist for Nerves/Anxiety/Tension/Depression"="Seen A Psychiatrist For Nerves, Anxiety, Tension Or Depression")
bin_matrix_df<-rename(bin_matrix_df,"Health Rating (self-reported)"="Overall Health Rating")
bin_matrix_df<-rename(bin_matrix_df,"Health Score (IMD calculated)"="Health Score")
bin_matrix_df<-rename(bin_matrix_df,"Respiratory & Allergy Conditions (Doctor-diagnosed)"="Blood Clot, DVT, Bronchitis, Emphysema, Asthma, Rhinitis, Eczema, Allergy Diagnosed By Doctor")

#bin_matrix_df<-rename(bin_matrix_df,"Long worried feeling after embarassing experience"="Worry Too Long After Embarrassment")
#bin_matrix_df<-rename(bin_matrix_df,"Fresh Fruits intake"="Fresh Fruits")
#bin_matrix_df<-rename(bin_matrix_df,"Moderate physical activity"="Moderate Phys Act")
#bin_matrix_df<-rename(bin_matrix_df,"Vigorous physical activity"="Vigorous Phys Act")
#bin_matrix_df<-rename(bin_matrix_df,"Morning or Evening person"="Morning Evening Person")

selected_features <- colnames(bin_matrix_df)

#Remove overlapping smoking variables
smoke_del <- c("Current Smok","Ever Smoked")
selected_features<-selected_features[! selected_features %in% smoke_del ]


#CHECK MODELS
# for (e in models){
#   bin_matrix_df[[e]] <- ifelse(rownames(bin_matrix)==e,1,0)
# }

# Create the duplicated binary matrix
duplicated_bin_matrix <- do.call(rbind, lapply(1:8, function(i) bin_matrix_df[rep(i, 9 - i +100), ]))
duplicated_bin_matrix[selected_features] <- lapply(duplicated_bin_matrix[selected_features], as.logical)

list_anxiety_var <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Variables_blocks/list_anxiety_var.rds")
list_anxiety_var <- gsub("\\.0\\.0$", "",list_anxiety_var)  # Remove "0.0" at the end
list_anxiety_var<- gsub("_", " ", list_anxiety_var)         # Replace underscores with spaces

# Capitalize the first letter of each word
list_anxiety_var <- sapply(strsplit(list_anxiety_var, " "), function(x) {
  paste(toupper(substring(x, 1, 1)), substring(x, 2), sep = "", collapse = " ")
})

anx_var_selected_features<- selected_features[selected_features  %in% list_anxiety_var ]
anx_var_selected_features <- c(anx_var_selected_features,"Seen GP for Nerves/Anxiety/Tension/Depression","Seen Psychiatrist for Nerves/Anxiety/Tension/Depression")
colour_metadata = data.frame(
  set=selected_features,
  color_var=ifelse(selected_features %in% anx_var_selected_features, 'Anxiety',
                   ifelse(selected_features %in% c("Sleeplessness / Insomnia","Long-standing Illness, Disability Or Infirmity", "Health Score (IMD calculated)","Health Rating (self-reported)","Cancer By Doctor","Respiratory & Allergy Conditions (Doctor-diagnosed)"), 'Global Health',
                          ifelse(selected_features %in% c("Pack Years Of Smoking","Vigorous Phys Act","Water Intake","Smokers In Household","Alcohol Drinker Status","Smoking Status","Morning Evening Person"),'Lifestyle',
                                 ifelse(selected_features %in% c("Job Involves Shift Work","Income Score","Education Score","Owner Or Rent","Current Employment Status"),'Socio-economic',
                                        ' ')
))))

models <- c("Lasso & ElNet Male", "RF Male","XGb Male","Lasso Female", "ElNet Female", "RF Female", "XGb Female")

#########
annotation_df <- data.frame(
  x = seq(2, length(models),1),  # Adjust x-position as needed
  y = 0,  # Set y positions
  label = models[2:length(models)]
)
annotation_df<- rbind(annotation_df,data.frame(
  x = c(0.75,1.1),  # Adjust x-position as needed
  y = c(0,0),  # Set y positions
  label = c("Lasso Male" ,"ElNet Male")
))
  
rect_df <- data.frame(
  xmin = c(0.5,3.5),  # Starting x-coordinate for each rectangle
  xmax = c(3.5,7.5),  # Ending x-coordinate for each rectangle
  ymin = 0.5,  # Extend to the bottom
  ymax = length(selected_features)+0.5,   # Extend to the top
  fill = c("#0000CCFF","white")  # Color for each rectangle
)


p <- upset(
  duplicated_bin_matrix,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  set_sizes=(
    upset_set_size()
    + theme(axis.text.x=element_text(size = 12),
    )),
  stripes=upset_stripes(
    mapping=aes(color=color_var),
    geom = geom_segment(size = 5),
    colors=c(
      'Anxiety'='darkgrey',
      'Global Health'='#5E9B8B',
      'Lifestyle'='#4682B4',
      'Socio-economic'="#C95D30",
      ' '='white'
    ),
    data=colour_metadata
  )
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 4, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    alpha = 0.4 ,
    inherit.aes = FALSE,
    fill = rect_df$fill,  # Set the fill directly
  ) +
  guides(fill = "none")  #Remove the fill legend
p <- p + 
  theme(
    plot.margin = margin(20, 5, 30, 5),
    axis.text.y = element_text(size = 12, face = "bold"),  # Set size text
    text = element_text(size = 14)  # General font increase
  )
print(p)
ggsave('upset_updated_2.png', plot = p, width = 14, height = 10, dpi = 300, bg="transparent")

p <- upset(
  bin_matrix_df,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  set_sizes=(
    upset_set_size()
    + theme(axis.text.x=element_text(size = 12),
    )),
  stripes=c('white')
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 10, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    alpha = 0.5 ,
    inherit.aes = FALSE,
    fill = rect_df$fill,  # Set the fill directly
  ) +
  guides(fill = "none")  #Remove the fill legend
p <- p + 
  theme(
    plot.margin = margin(20, 5, 30, 5),
    axis.text.y = element_text(size = 14, face = "bold"),  # Set size text
    text = element_text(size = 16)  # General font increase
  )

print(p)
ggsave('upset_scale_size_.png', plot = p, width = 14, height = 10, dpi = 300, bg="transparent")

#### BIOMARKERS

#CREATION OF THE MATRIX
X_train_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_male_bio.rds")
features_male <- colnames(X_train_male)
X_train_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_fem_bio.rds")
features_female <- colnames(X_train_female)

models <- c("Lasso Female","Lasso Male","RF Female", "RF Male")

# Create binary presence/absence matrix
binary_matrix <- data.frame(matrix(0, nrow=length(models), ncol=length(features_female)))
colnames(binary_matrix) <- features_female
rownames(binary_matrix) <- models

top_n <- 20
path_RF <- "/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers/"

# Random Forest Male
importances_RF_male <- fromJSON(file.path(path_RF, "importances_var_male.json"))
indices_RF_male <- order(importances_RF_male, decreasing = TRUE)[1:top_n]
top_features_RF_male <- features_male[indices_RF_male]
binary_matrix["RF Male", top_features_RF_male] <- 1

# Random Forest Female
importances_RF_female <- fromJSON(file.path(path_RF, "importances_var_female.json"))
indices_RF_female <- order(importances_RF_female, decreasing = TRUE)[1:top_n]
top_features_RF_female <- features_female[indices_RF_female]
binary_matrix["RF Female", top_features_RF_female] <- 1


# LASSO Male
variables_selection_LASSO_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stability.selection_bio_male.rds")
variables_selection_LASSO_male_names <- variables_selection_LASSO_male$rowname
# Add specific variables
binary_matrix["Lasso Male", variables_selection_LASSO_male_names] <- 1


# LASSO Female
variables_selection_LASSO_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stability.selection_bio_fem.rds")
variables_selection_LASSO_female_names <- variables_selection_LASSO_female$rowname
binary_matrix["Lasso Female", variables_selection_LASSO_female_names] <- 1


# Drop columns with only zeros
binary_matrix <- binary_matrix[, colSums(binary_matrix) > 0]
binary_matrix[is.na(binary_matrix)] <- 0

saveRDS(binary_matrix,"/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female_bio.rds")

#CREATION OF THE PLOT
bin_matrix <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female_bio.rds")
models <- c("Lasso Female","Lasso Male","RF Female", "RF Male")
sex_indicator <- c("Female", "Male", "Female", "Male")

bin_matrix_df <- as.data.frame(bin_matrix)
bin_matrix_df<-bin_matrix_df[,colnames(bin_matrix_df)[colSums(bin_matrix_df)!=1]]
colnames(bin_matrix_df) <- gsub("\\.0\\.0$", "", colnames(bin_matrix_df))  # Remove "0.0" at the end
colnames(bin_matrix_df) <- gsub("_", " ", colnames(bin_matrix_df))               # Replace underscores with spaces

# Capitalize the first letter of each word
colnames(bin_matrix_df) <- sapply(strsplit(colnames(bin_matrix_df), " "), function(x) {
  paste(toupper(substring(x, 1, 1)), substring(x, 2), sep = "", collapse = " ")
})

# Convert to a vector
colnames(bin_matrix_df) <- as.vector(colnames(bin_matrix_df))

#CHECK MODELS
# for (e in models){
#   bin_matrix_df[[e]] <- ifelse(rownames(bin_matrix)==e,1,0)
# }
selected_features <- colnames(bin_matrix_df)

# Create the duplicated binary matrix
duplicated_bin_matrix <- do.call(rbind, lapply(1:4, function(i) bin_matrix_df[rep(i, 5 - i +100), ]))
duplicated_bin_matrix[selected_features] <- lapply(duplicated_bin_matrix[selected_features], as.logical)

#########
annotation_df <- data.frame(
  x = seq(1, length(models),1),  # Adjust x-position as needed
  y = 0,  # Set y positions
  label = models
)
colors <- ifelse(sex_indicator=="Male","#0000CCFF","white")
rect_df <- data.frame(
  xmin = seq(0.5, 3.5, 1),  # Starting x-coordinate for each rectangle
  xmax = seq(1.5, 4.5, 1),  # Ending x-coordinate for each rectangle
  ymin = 0.5,  # Extend to the bottom
  ymax = length(selected_features)+0.5,   # Extend to the top
  fill = colors  # Color for each rectangle
)


p <- upset(
  duplicated_bin_matrix,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  stripes=c('white')
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 3, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = rect_df$fill,
    alpha = 0.5 ,
    inherit.aes = FALSE
  ) +
  guides(fill = "none")  #Remove the fill legend
p <- p + theme(plot.margin = margin(20, 5, 30, 5))  # Adjust bottom margin (third value)

print(p)
ggsave('upset_bio.png', plot = p, width = 10, height = 7, dpi = 300, bg="transparent")

p <- upset(
  bin_matrix_df,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  stripes=c('white')
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 4, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = rect_df$fill,
    alpha = 0.25 ,
    inherit.aes = FALSE
  ) +
  guides(fill = "none")  #Remove the fill legend
p <- p + theme(plot.margin = margin(20, 5, 30, 5))  # Adjust bottom margin (third value)

print(p)
ggsave('upset_scale_size_bio.png', plot = p, width = 10, height = 7, dpi = 300, bg="transparent")

####

#### Non-Anxiety VAR

#CREATION OF THE MATRIX
X_train_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_male_nobio_RF.rds")
features_male <- colnames(X_train_male)
X_train_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_fem_nobio_RF.rds")
features_female <- colnames(X_train_female)

models <- c("Lasso Female","Lasso Male","RF Female", "RF Male")

# Create binary presence/absence matrix
binary_matrix <- data.frame(matrix(0, nrow=length(models), ncol=length(features_female)))
colnames(binary_matrix) <- features_female
rownames(binary_matrix) <- models

top_n <- 20
path_RF <- "/rds/general/project/hda_24-25/live/TDS/Group03/RF/Without_anxiety_var/"

# Random Forest Male
importances_RF_male <- fromJSON(file.path(path_RF, "importances_var_male.json"))
indices_RF_male <- order(importances_RF_male, decreasing = TRUE)[1:top_n]
top_features_RF_male <- features_male[indices_RF_male]
binary_matrix["RF Male", top_features_RF_male] <- 1

# Random Forest Female
importances_RF_female <- fromJSON(file.path(path_RF, "importances_var_female.json"))
indices_RF_female <- order(importances_RF_female, decreasing = TRUE)[1:top_n]
top_features_RF_female <- features_female[indices_RF_female]
binary_matrix["RF Female", top_features_RF_female] <- 1

# LASSO Male
variables_selection_LASSO_male <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stability.selection_lasso_male_non_anx.rds")
variables_selection_LASSO_male_names <- names(variables_selection_LASSO_male)
# Filter out variables containing "0.0"
non_0_var <- variables_selection_LASSO_male_names[!grepl("0.0", variables_selection_LASSO_male_names)]
print(non_0_var)
variables_selection_LASSO_male_names <- unique(sub("0.0.*", "0.0", variables_selection_LASSO_male_names[!(variables_selection_LASSO_male_names %in% non_0_var)]))
# Add specific variables
variables_selection_LASSO_male_names <- c(variables_selection_LASSO_male_names, "health_score", "Overall_health_rating")
binary_matrix["Lasso Male", variables_selection_LASSO_male_names] <- 1



# LASSO Female
variables_selection_LASSO_female <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/stability.selection_lasso_fem_non_anx.rds")
variables_selection_LASSO_female_names <- names(variables_selection_LASSO_female)
non_0_var <- variables_selection_LASSO_female_names[!grepl("0.0", variables_selection_LASSO_female_names)]
print(non_0_var)
variables_selection_LASSO_female_names <- unique(sub("0.0.*", "0.0", variables_selection_LASSO_female_names[!(variables_selection_LASSO_female_names %in% non_0_var)]))
variables_selection_LASSO_female_names <- c(variables_selection_LASSO_female_names, "health_score", "Overall_health_rating", "income_score")
binary_matrix["Lasso Female", variables_selection_LASSO_female_names] <- 1


# Drop columns with only zeros
binary_matrix <- binary_matrix[, colSums(binary_matrix) > 0]
binary_matrix[is.na(binary_matrix)] <- 0

saveRDS(binary_matrix,"/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female_non_anx.rds")

#CREATION OF THE PLOT
bin_matrix <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Binary_matrix_Male_Female_non_anx.rds")
models <- c("Lasso Female","Lasso Male","RF Female", "RF Male")
sex_indicator <- c("Female", "Male", "Female", "Male")

bin_matrix_df <- as.data.frame(bin_matrix)
bin_matrix_df<-bin_matrix_df[,colnames(bin_matrix_df)[colSums(bin_matrix_df)!=1]]
colnames(bin_matrix_df) <- gsub("\\.0\\.0$", "", colnames(bin_matrix_df))  # Remove "0.0" at the end
colnames(bin_matrix_df) <- gsub("_", " ", colnames(bin_matrix_df))               # Replace underscores with spaces

# Capitalize the first letter of each word
colnames(bin_matrix_df) <- sapply(strsplit(colnames(bin_matrix_df), " "), function(x) {
  paste(toupper(substring(x, 1, 1)), substring(x, 2), sep = "", collapse = " ")
})

# Convert to a vector
colnames(bin_matrix_df) <- as.vector(colnames(bin_matrix_df))

selected_features <- colnames(bin_matrix_df)

# Create the duplicated binary matrix
duplicated_bin_matrix <- do.call(rbind, lapply(1:4, function(i) bin_matrix_df[rep(i, 5 - i +100), ]))
duplicated_bin_matrix[selected_features] <- lapply(duplicated_bin_matrix[selected_features], as.logical)

#########
annotation_df <- data.frame(
  x = seq(1, length(models),1),  # Adjust x-position as needed
  y = 0,  # Set y positions
  label = models
)
colors <- ifelse(sex_indicator=="Male","red","blue")
rect_df <- data.frame(
  xmin = seq(0.5, 3.5, 1),  # Starting x-coordinate for each rectangle
  xmax = seq(1.5, 4.5, 1),  # Ending x-coordinate for each rectangle
  ymin = 0.5,  # Extend to the bottom
  ymax = length(selected_features)+0.5,   # Extend to the top
  fill = colors  # Color for each rectangle
)

p <- upset(
  duplicated_bin_matrix,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  stripes=c('white')
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 3, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
    alpha = 0.25 ,
    inherit.aes = FALSE
  ) +
  guides(fill = "none")  #Remove the fill legend

print(p)
ggsave('upset_non_anx_var.png', plot = p, width = 10, height = 7, dpi = 300, bg="transparent")

p <- upset(
  bin_matrix_df,
  selected_features,
  name = "",
  sort_intersections="descending",
  # sort_sets="descending",
  base_annotations = list(),
  stripes=c('white')
)+
  geom_text(
    data = annotation_df, 
    aes(x = x, y = y, label = label),
    size = 4, 
    hjust = 0, 
    color = "black", 
    angle = -50
  )+  # Transparent colored rectangles
  coord_cartesian(clip = "off") + 
  geom_rect(
    data = rect_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
    alpha = 0.25 ,
    inherit.aes = FALSE
  ) +
  guides(fill = "none")  #Remove the fill legend

print(p)
ggsave('upset_scale_size_non_anx_var.png', plot = p, width = 10, height = 7, dpi = 300, bg="transparent")

####