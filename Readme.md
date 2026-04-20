# Introduction
This repository contains the code for the research paper: ---

Even though the research paper makes use of a .sqlite3 database, the code is made to work for data within a .csv as well, with specific database functions living in seperate classes.
As you can find in the paper the pipeline itself consists of the following steps:
1. preprocess.py -> Preprocessing, translating, and filtering 
2. classify.py -> Few-shot classification
3. analysis.py -> Downstream analyses and graphs

The pipeline is supported by two seperate parts:
1. database.py -> contains generic methods for extracting specific data from the .sqlite3 database
2. validation.py -> contains methods to aid with validating and testing the performance of the few-shot classification 

## General use tips
**The most easy way to use the code, is to clone the repository with git.**

Most classes take specific path names for the files they might use. As a default this path mirrors the repository of the code by making use of the parent directory, i.e. the directory that contains the subdirectory 'src' 
If you wish to change the file structure of the code, be sure to change the path arguments for the classes. 

**Be sure to also copy the following file: src/channel_def.json -> This contains the definitions for official/unofficial channel and the ordering with which they entire the multiverse analysis.**

If you cloned the repository, this should be included. 

**If you use the code in combination with a .sqlite3 database, it has to follow the structure desired by the code.**

database.py has an enforce_structure() method, but this is only useful for adding the classification specific columns in an existing table which consists of ids, video_ids, language_ids, and subtitles. 
You can off course also adapt the code to your own databases' structure. 

# Preprocessing, translating, and filtering
**Initialize the class Preprocess()**

It does not require any arguments, but initializes a path for the translation model within models/. It is advised to change this path if your structure differs
Within init() you can also find some hyperparameters for the filtering and the translation batches. You can change these values as you please by inputting it while initializing the class in your code. 
Finally, the default device for the translation is 'cuda', if you plan to run a translation but do not have an (nvidia powered) GPU, it is advised to change it to 'cpu' or some other device.

**Run Preprocess.preprocess_text**

This method takes a DataFrame and requires a string to indicate the column name for the transcripts.
It will remove standard watermarks and timestamps provided by TikTok for their transcripts. Moreover, it will parse any transcripts delivered in .json format. 
It returns a DataFrame with the filtered data in the original column (it overwrites the old text for the new DataFrame)

**Run Preprocess.generate_translation()**

This method takes a DataFrame with two strings, indicating the column name for the transcripts and the column name for the language_ids.
The language_ids are specific to the database for the present research. However, if you have an integer that indicates the language of the transcript, you can adapt the numbers 2 and 18 to fit your prefered languages and structure. Be sure to also change the input arguments to the model (src_lang, tgt_lang)
Currently, it only translates Dutch transcripts (id: 18) and leaves English transcripts (id: 2) as is. Any other language_ids get a 'OTHER LANG' label
It return a DataFrame with the translated transcripts in the column 'translation'

**Run Preprocess.filter_on_heuristics()**

This method takes a DataFrame and a string, which indicates the name of the column for the transcripts. 
It returns a DataFrame with a 'filtered' column (boolean) that indicates whether the transcript in that row is filtered (1) or not (0)

# Few-shot classification
This script contains two small methods to be used in tandem with the ChoiceLLM package: https://github.com/mwestera/choicellm

**Run transcripts_to_txt() before ChoiceLLM**

This method takes a DataFrame, a string to indicate the transcript column name, a string to indicate the description element column name and a string to indicate the target to merge on (e.g. id, video_id)
The method will create two files: 'transcripts.txt' and id.csv
transcripts.txt can be fed as an argument to ChoiceLLM and contains on each row the text transcript and the description elements. ChoiceLLM in combination with a language model will then classify each row of this .txt
id.csv is made to ensure you can merge your results back into the database and consists of ids that belong to the transcripts in transcripts.txt

**Run get_results() after ChoiceLLM**

This method takes strings to indicate: the name of the results file, the name of the id file, and the column names for the results (depends on which results you want from ChoiceLLM -> See their GitHub)
ChoiceLLM will create a 'results.csv' in your directory. This contains the results of the classification on 'transcripts.txt'
The method will then take id.csv and merge to results to it. 
It contains an assert clause to trigger and halt the code when id.csv and transcripts.txt are not of the same length. This can indicate the something went wrong and the transcripts are likely not aligned to their original ids. 

*To see this entire pipeline in action, check the classify.slurm script. This is a bash script (among others) that runs the classification on a HPC*
