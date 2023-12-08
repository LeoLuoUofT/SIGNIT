WARNING: The code is fully functional on my machine however due to us not having access to fresh machines to test a raw implementation on (since we can't seem to install the files we need on the lab computers, i.e. we cannot change enviornment variables on cluster), you may run into errors when running it on the cluster.


You must install https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.12.03 and set the bin on your path along with a compatbile version of 'pyarrow' and 'fastparquet' for your machine. The reason we cannot simply pip install ytdl is because the Hamburg Regional Court in Germany has taken the main package and removed some of its connecting API causing the pip version to not function: https://yt-dl.org/downloads.

Then simply run:

pip install requirements.txt
python main.py

If you also wanted to see the intermediate images, change the no_sanity variable in no_sanity.py to TRUE.

This will run the main part of the code but if you wanted to create your own training/testing data from the ASL Kaggle Alphabet Library. You can navigate to Leo_testing_folder/dataset_creation. The dataset_create.py will take every nth image from every subfolder and put them into a new folder (where n is your first input), once you've made your training and test sets use filterouttrainingimages.py to remove any images in the training set that occur in the testing set. A bash file is set up to preform this for you if you need it.

If you wanted to see the outputs for multiple livestreams our stream_input.py supports that, just uncomment the last section and input the streams you wish to process.

You also may run into problems with tensorflow depending on your operating system, this is because we are using an older version of tensorflow. If this is the case you will need to uninstall and reinstall the correct version and reconfigure your enviornment variables.

 The reason the tensorflow version in requirements.txt is not the most recent one is mainly because tensorflow support for a Windows native GPU seems to have been discontinued beyond this version. If you're running windows you may need to follow this guide: https://www.tensorflow.org/install/pip#windows-native.


To run the Classifier:-

a)Download the csv files needed for model training and testing from the below link:- 
https://drive.google.com/drive/folders/1Wkrkvy6Bl-JNvMFeZDGc8utKjkS7LJXt?usp=sharing

b)Now upload these csv files onto google colab and run the classifier.ipynb file present in the visualizations folder of the repository on google colab with these csv.

c)Sit back, relax and wait for results.


More visualization and data analysis stats is provided in the Visualizations folder, included the jupyter notebook for analysis on Deaf users and sign_langauge_users_analysis.py for analysis on deaf users who use sign language.
