import os
import shutil
import random

TRAIN_SIZE = 500



def create_test_folder(image_folder=r"data\Animal Images", random_seed=42):
    # Create the test folder if it doesn't exist
    
    num_samples = TRAIN_SIZE

    train_path = os.path.join(image_folder, "train")
    test_path = os.path.join(image_folder, "test")
    
    
    

    subfolders = [os.path.join(train_path, subfolder) for subfolder in os.listdir(train_path)]

    for folder in subfolders:
        files = []
        print(folder)
    
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))



        # Ensure num_samples_per_folder does not exceed the number of files
        num_samples = min(num_samples, len(files))
        random.seed(random_seed)
        # Randomly sample files
        files = random.sample(files, num_samples)
        #sampled_dogs = random.sample(files, num_samples)

        # Copy sampled files to the test folder and delete the originals
        for file_path in files:
            _, file_name = os.path.split(file_path)
            print(file_path)
            test_file = ""
            if os.path.basename(folder)=="cats":
                test_file = os.path.join(test_path+r"\cats", file_name)
            elif os.path.basename(folder)=="dogs":
                test_file = os.path.join(test_path+r"\dogs", file_name)
            shutil.copyfile(file_path, test_file)
            os.remove(file_path)

create_test_folder()