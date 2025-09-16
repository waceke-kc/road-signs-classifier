

# Create virtual environment 
    python -m venv nanodet_env 
    source nanodet_env/bin/activate 
# On Windows: nanodet_env\Scripts\activate 
# Install dependencies 
pip install -r requirements.txt 
# Install the nanodet module downloaded
    cd nanodet 
    pip install -e .

### Create the dataset

I am using the main directory to run the code but feel free to change the paths depending on your starting point.  In order to create the dataset, I used the following command

    python3 code/dataset_preparation.py data/datasets/dataset_original/ data/dataset_test/ --create_coco

### Train the model

The first thing to do is to duplicate the **nanodet-plus-m-1.5x_320.yml8** config yml file and rename it to my_dataset and then save it in nanodet/config.

Then I changed the necessary configs like the location of the train and validation dataset, number of classes etc...

Then I ran the training script using
    python3 nanodet/tools/train.py nanodet/config/my_dataset.yml 

### Video Editor
Once the training and testing is done, you can use the video editor to test the model.
Use the model found in the workspace folder.
        python3 code/NanoDETVideoPlayer.py 


