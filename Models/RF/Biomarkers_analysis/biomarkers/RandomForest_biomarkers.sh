#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=20:mem=20gb
#PBS -N RF_female_bio

cd /rds/general/user/tt1024/projects/hda_24-25/live/TDS/Group03/RF/biomarkers

module load anaconda3/personal
source activate TDS_pyp1

python RandomForest_biomarkers_female.py
