library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggtext)

roc_curve_RF_male <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/roc_curve_RF_male.txt", 
                                 header = TRUE, sep = ",",quote="")
auc_RF_male <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/auc_RF_male.txt", 
                           header = TRUE, sep = ",",quote="")
auc_RF_male <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_male)[1]))
roc_curve_RF_male <- roc_curve_RF_male %>% mutate(Model = "RF", Sex = "All variables", AUC= auc_RF_male)

roc_curve_RF_male_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/Without_anxiety_var/roc_curve_RF_male.txt", 
                               header = TRUE, sep = ",",quote="")
auc_RF_male_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/Without_anxiety_var/auc_RF_male.txt", 
                         header = TRUE, sep = ",",quote="")
auc_RF_male_var_no_anx <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_male_var_no_anx)[1]))
roc_curve_RF_male_var_no_anx <- roc_curve_RF_male_var_no_anx %>% mutate(Model = "RF", Sex = "No Anxiety variables", AUC= auc_RF_male_var_no_anx)

roc_curve_Lasso_male<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male.txt", 
                                 header = TRUE, sep = ",",quote="")
roc_curve_Lasso_male_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male_non_var_anx.txt", 
                                    header = TRUE, sep = ",",quote="")


roc_curve_Lasso_male <- roc_curve_Lasso_male %>% mutate(Model = "Lasso", Sex = "All variables")
roc_curve_Lasso_male <-roc_curve_Lasso_male %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )
roc_curve_Lasso_male_var_no_anx <- roc_curve_Lasso_male_var_no_anx %>% mutate(Model = "Lasso", Sex = "No Anxiety variables")
roc_curve_Lasso_male_var_no_anx <-roc_curve_Lasso_male_var_no_anx %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )
roc_data_males <- bind_rows(roc_curve_RF_male,roc_curve_RF_male_var_no_anx,roc_curve_Lasso_male,roc_curve_Lasso_male_var_no_anx)

roc_color_males <-c("#0000FF","#00FFFF", "#228B22","#7FFF00")
roc_color_females <-c("#A52A2A","#FF7F00", "#FF00FF","#7F00FF")
roc_anx <- c("#0000FF","#00FFFF","#FF0000","#7F00FF")
roc_no_var_anx <- c("#0000FF","#00FFFF","#A52A2A","#FF7F00")

# Precompute AUC labels and assign y-coordinates
auc_labels <- roc_data_males %>%
  group_by(Model, Sex) %>%
  summarise(AUC = unique(AUC), .groups = "drop") %>%
  #mutate(Label = paste(Model, "AUC:", round(AUC, 2)))
  mutate(Label = paste(Model, Sex, "AUC:", round(AUC, 2)))
auc_text <- auc_labels %>%
  #mutate(Label = paste0("AUC ",Model, " : ", round(AUC, 2) )) %>%
  mutate(Label = paste0("AUC ",Model, " ", Sex, " : ", round(AUC, 2) )) %>%
  pull(Label) %>%
  paste(collapse = "\n")  # Combine labels with line breaks


# Plot the ROC curves
g<- ggplot(roc_data_males, aes(x = FPR, y = TPR, color = interaction(Sex,Model))) +
  geom_line(linewidth = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +  
  scale_color_manual(values = roc_no_var_anx) +  # Apply manual color mapping
  labs(title = "ROC Curves for Males ",
       x = "False Positive Rate ",
       y = "True Positive Rate",
       color = "Legend") +
  theme_minimal(base_size = 16) +  # Increase the base font size
  theme(
    legend.text = element_markdown(size = 14),  # Increase legend text size
    legend.title = element_text(size = 16, face = "bold"),  # Legend title size
    axis.title = element_text(size = 14, face = "bold"),  # Axis labels
    axis.text = element_text(size = 14),  # Axis tick labels
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5)  # Centered title
  )  + geom_label(aes(x = -0.1, y = 0.9, label = auc_text, colour = unique(interaction(roc_data$Sex,roc_data$Model))), fill = "white", color = "black", size = 3.5, hjust = 0)

#print(g)

ggsave("/rds/general/project/hda_24-25/live/TDS/Group03/ROC_curve_anx_vars_removed_males.png",width = 10, height = 6, dpi = 300)


#Females
roc_curve_RF_female <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/roc_curve_RF_female.txt", 
                               header = TRUE, sep = ",",quote="")
auc_RF_female <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/auc_RF_female.txt", 
                         header = TRUE, sep = ",",quote="")
auc_RF_female <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_female)[1]))
roc_curve_RF_female <- roc_curve_RF_female %>% mutate(Model = "RF", Sex = "All variables", AUC= auc_RF_female)

roc_curve_RF_female_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/Without_anxiety_var/roc_curve_RF_female.txt", 
                                          header = TRUE, sep = ",",quote="")
auc_RF_female_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/Without_anxiety_var/auc_RF_female.txt", 
                                    header = TRUE, sep = ",",quote="")
auc_RF_female_var_no_anx <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_female_var_no_anx)[1]))
roc_curve_RF_female_var_no_anx <- roc_curve_RF_female_var_no_anx %>% mutate(Model = "RF", Sex = "No Anxiety variables", AUC= auc_RF_female_var_no_anx)

roc_curve_Lasso_female<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_fem.txt", 
                                 header = TRUE, sep = ",",quote="")
roc_curve_Lasso_female_var_no_anx <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_fem_non_var_anx.txt", 
                                             header = TRUE, sep = ",",quote="")


roc_curve_Lasso_female <- roc_curve_Lasso_female %>% mutate(Model = "Lasso", Sex = "All variables")
roc_curve_Lasso_female <-roc_curve_Lasso_female %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )
roc_curve_Lasso_female_var_no_anx <- roc_curve_Lasso_female_var_no_anx %>% mutate(Model = "Lasso", Sex = "No Anxiety variables")
roc_curve_Lasso_female_var_no_anx <-roc_curve_Lasso_female_var_no_anx %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )
roc_data_females <- bind_rows(roc_curve_RF_female,roc_curve_RF_female_var_no_anx,roc_curve_Lasso_female,roc_curve_Lasso_female_var_no_anx)

# Precompute AUC labels and assign y-coordinates
auc_labels <- roc_data_females %>%
  group_by(Model, Sex) %>%
  summarise(AUC = unique(AUC), .groups = "drop") %>%
  #mutate(Label = paste(Model, "AUC:", round(AUC, 2)))
  mutate(Label = paste(Model, Sex, "AUC:", round(AUC, 2)))
auc_text <- auc_labels %>%
  #mutate(Label = paste0("AUC ",Model, " : ", round(AUC, 2) )) %>%
  mutate(Label = paste0("AUC ",Model, " ", Sex, " : ", round(AUC, 2) )) %>%
  pull(Label) %>%
  paste(collapse = "\n")  # Combine labels with line breaks


# Plot the ROC curves
g<- ggplot(roc_data_females, aes(x = FPR, y = TPR, color = interaction(Sex,Model))) +
  geom_line(linewidth = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +  
  scale_color_manual(values = roc_no_var_anx) +  # Apply manual color mapping
  labs(title = "ROC Curves for Females ",
       x = "False Positive Rate ",
       y = "True Positive Rate",
       color = "Legend") +
  theme_minimal(base_size = 16) +  # Increase the base font size
  theme(
    legend.text = element_markdown(size = 14),  # Increase legend text size
    legend.title = element_text(size = 16, face = "bold"),  # Legend title size
    axis.title = element_text(size = 14, face = "bold"),  # Axis labels
    axis.text = element_text(size = 14),  # Axis tick labels
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5)  # Centered title
  )  + geom_label(aes(x = -0.1, y = 0.9, label = auc_text, colour = unique(interaction(roc_data$Sex,roc_data$Model))), fill = "white", color = "black", size = 3.5, hjust = 0)

#print(g)

ggsave("/rds/general/project/hda_24-25/live/TDS/Group03/ROC_curve_anx_vars_removed_females.png",width = 10, height = 6, dpi = 300)



####Biomarkers sensitivity analysis

#MALE

#RF
roc_curve_RF_male_all <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/roc_curve_RF_male.txt", 
                                   header = TRUE, sep = ",",quote="")
auc_RF_male_all<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/auc_RF_male.txt", 
                             header = TRUE, sep = ",",quote="")
auc_RF_male_all <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_male_all)[1]))
roc_curve_RF_male_all <- roc_curve_RF_male_all %>% mutate(Model = "RF All", Sex = "Male", AUC= auc_RF_male_all)

roc_curve_RF_male_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers/roc_curve_RF_male.txt", 
                                   header = TRUE, sep = ",",quote="")
auc_RF_male_bio<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers/auc_RF_male.txt", 
                            header = TRUE, sep = ",",quote="")
auc_RF_male_bio <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_male_bio)[1]))
roc_curve_RF_male_bio <- roc_curve_RF_male_bio %>% mutate(Model = "RF biomarkers", Sex = "Male", AUC= auc_RF_male_bio)


roc_curve_RF_male_no_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/no_biomarkers/roc_curve_RF_male.txt", 
                                   header = TRUE, sep = ",",quote="")
auc_RF_male_no_bio<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/no_biomarkers/auc_RF_male.txt", 
                            header = TRUE, sep = ",",quote="")
auc_RF_male_no_bio <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_male_no_bio)[1]))
roc_curve_RF_male_no_bio <- roc_curve_RF_male_no_bio %>% mutate(Model = "RF All but biomarkers ", Sex = "Male", AUC= auc_RF_male_no_bio)

#LASSO
roc_curve_Lasso_male_all <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_male_all_variables_T.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_male_all <- roc_curve_Lasso_male_all %>% mutate(Model = "Lasso All", Sex = "Male")
roc_curve_Lasso_male_all <-roc_curve_Lasso_male_all %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )

roc_curve_Lasso_male_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male_bio.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_male_bio <- roc_curve_Lasso_male_bio %>% mutate(Model = "Lasso Biomarkers", Sex = "Male")
roc_curve_Lasso_male_bio <-roc_curve_Lasso_male_bio %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )

roc_curve_Lasso_male_no_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_male_all_variables_but_biomarkers.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_male_no_bio <- roc_curve_Lasso_male_no_bio %>% mutate(Model = "Lasso All but biomarkers", Sex = "Male")
roc_curve_Lasso_male_no_bio <-roc_curve_Lasso_male_no_bio %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )


roc_color_Lasso <-c("#0000FF","#758bd1","#75b8d1")
roc_color_RF <-c("#A52A2A","#FF7F00", "#d1ab75")

roc_data_male <- bind_rows(roc_curve_RF_male_all,roc_curve_RF_male_bio,roc_curve_RF_male_no_bio,roc_curve_Lasso_male_all,roc_curve_Lasso_male_bio, roc_curve_Lasso_male_no_bio)

# Precompute AUC labels and assign y-coordinates
auc_labels <- roc_data_male %>%
  group_by(Model, Sex) %>%
  summarise(AUC = unique(AUC), .groups = "drop") %>%
  mutate(Label = paste(Model, "AUC:", round(AUC, 2)))
  #mutate(Label = paste(Model, Sex, "AUC:", round(AUC, 2)))
auc_text <- auc_labels %>%
  mutate(Label = paste0("AUC ",Model, " : ", round(AUC, 2) )) %>%
  #mutate(Label = paste0("AUC ",Model, " ", Sex, " : ", round(AUC, 2) )) %>%
  pull(Label) %>%
  paste(collapse = "\n")  # Combine labels with line breaks


# Plot the ROC curves
g<- ggplot(roc_data_male, aes(x = FPR, y = TPR, color = Model,linetype = Model)) +
  geom_line(linewidth = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +  
  scale_color_manual(values = c(roc_color_RF, roc_color_Lasso)) +  # Apply manual color mapping
  scale_linetype_manual(values =c("solid", "dashed","dotted","solid","dashed","dotted")) + #Apply manual shape
  labs(title = "ROC Curves for Males",
       x = "False Positive Rate ",
       y = "True Positive Rate") +
  theme_minimal(base_size = 16) +  # Increase the base font size
  theme(
    legend.text = element_markdown(size = 14),  # Increase legend text size
    legend.title = element_text(size = 16, face = "bold"),  # Legend title size
    axis.title = element_text(size = 14, face = "bold"),  # Axis labels
    axis.text = element_text(size = 14),  # Axis tick labels
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5)  # Centered title
  )  + geom_label(aes(x = -0.02, y = 0.9, label = auc_text, colour = unique(interaction(roc_data$Sex,roc_data$Model))), fill = "white", color = "black", size = 3.4, hjust = 0)

print(g)
ggsave("/rds/general/project/hda_24-25/live/TDS/Group03/ROC_biomarkers_SA_males.png",width = 10, height = 6, dpi = 300)



#FEMALE

#rf
roc_curve_RF_female_all<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/roc_curve_RF_female.txt", 
                                    header = TRUE, sep = ",",quote="")
auc_RF_female_all <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/auc_RF_female.txt", 
                               header = TRUE, sep = ",",quote="")
auc_RF_female_all <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_female_all)[1]))
roc_curve_RF_female_all <- roc_curve_RF_female_all %>% mutate(Model = "RF All", Sex = "Female", AUC= auc_RF_female_all)

roc_curve_RF_female_bio<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers/roc_curve_RF_female.txt", 
                                    header = TRUE, sep = ",",quote="")
auc_RF_female_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers/auc_RF_female.txt", 
                               header = TRUE, sep = ",",quote="")
auc_RF_female_bio <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_female_bio)[1]))
roc_curve_RF_female_bio <- roc_curve_RF_female_bio %>% mutate(Model = "RF Biomarkers ", Sex = "Female", AUC= auc_RF_female_bio)

roc_curve_RF_female_no_bio<-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/no_biomarkers/roc_curve_RF_female.txt", 
                                       header = TRUE, sep = ",",quote="")
auc_RF_female_no_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/RF/biomarkers_and_all/no_biomarkers/auc_RF_female.txt", 
                                  header = TRUE, sep = ",",quote="")
auc_RF_female_no_bio <- as.numeric(gsub("AUC\\.\\.", "", colnames(auc_RF_female_no_bio)[1]))
roc_curve_RF_female_no_bio <- roc_curve_RF_female_no_bio %>% mutate(Model = "RF All but biomarkers", Sex = "Female", AUC= auc_RF_female_no_bio)

#LASSO
roc_curve_Lasso_fem_all <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_fem_all_variables_T.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_fem_all <- roc_curve_Lasso_fem_all %>% mutate(Model = "Lasso All", Sex = "Female")
roc_curve_Lasso_fem_all <-roc_curve_Lasso_fem_all %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )

roc_curve_Lasso_fem_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_fem_bio.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_fem_bio <- roc_curve_Lasso_fem_bio %>% mutate(Model = "Lasso Biomarkers", Sex = "Female")
roc_curve_Lasso_fem_bio <-roc_curve_Lasso_fem_bio %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )

roc_curve_Lasso_fem_no_bio <-read.delim("/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_fem_all_variables_but_biomarkers.txt", header = TRUE, sep = ",",quote="")
roc_curve_Lasso_fem_no_bio <- roc_curve_Lasso_fem_no_bio %>% mutate(Model = "Lasso All but biomarkers", Sex = "Female")
roc_curve_Lasso_fem_no_bio <-roc_curve_Lasso_fem_no_bio %>% 
  rename(AUC = X.AUC.,
         FPR = X.FPR.,
         TPR = X.TPR.
  )

roc_color_Lasso <-c("#0000FF","#758bd1","#75b8d1")
roc_color_RF <-c("#A52A2A","#FF7F00", "#d1ab75")

roc_data_female <- bind_rows(roc_curve_RF_female_all,roc_curve_RF_female_bio,roc_curve_RF_female_no_bio,roc_curve_Lasso_fem_all,roc_curve_Lasso_fem_bio,roc_curve_Lasso_fem_no_bio)

# Precompute AUC labels and assign y-coordinates
auc_labels <- roc_data_female %>%
  group_by(Model, Sex) %>%
  summarise(AUC = unique(AUC), .groups = "drop") %>%
  mutate(Label = paste(Model, "AUC:", round(AUC, 2)))
#mutate(Label = paste(Model, Sex, "AUC:", round(AUC, 2)))
auc_text <- auc_labels %>%
  mutate(Label = paste0("AUC ",Model, " : ", round(AUC, 2) )) %>%
  #mutate(Label = paste0("AUC ",Model, " ", Sex, " : ", round(AUC, 2) )) %>%
  pull(Label) %>%
  paste(collapse = "\n")  # Combine labels with line breaks


# Plot the ROC curves
g<- ggplot(roc_data_female, aes(x = FPR, y = TPR, color = Model,linetype = Model)) +
  geom_line(linewidth = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +  
  scale_color_manual(values = c(roc_color_RF, roc_color_Lasso)) +  # Apply manual color mapping
  scale_linetype_manual(values =c("solid", "dashed","dotted","solid","dashed","dotted")) + #Apply manual shape
  labs(title = "ROC Curves for Females",
       x = "False Positive Rate ",
       y = "True Positive Rate") +
  theme_minimal(base_size = 16) +  # Increase the base font size
  theme(
    legend.text = element_markdown(size = 14),  # Increase legend text size
    legend.title = element_text(size = 16, face = "bold"),  # Legend title size
    axis.title = element_text(size = 14, face = "bold"),  # Axis labels
    axis.text = element_text(size = 14),  # Axis tick labels
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5)  # Centered title
  )  + geom_label(aes(x = -0.02, y = 0.9, label = auc_text, colour = unique(interaction(roc_data$Sex,roc_data$Model))), fill = "white", color = "black", size = 3.4, hjust = 0)

print(g)
ggsave("/rds/general/project/hda_24-25/live/TDS/Group03/ROC_biomarkers_SA_females.png",width = 10, height = 6, dpi = 300)

