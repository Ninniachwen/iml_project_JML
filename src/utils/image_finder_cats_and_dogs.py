import os

LABEL ={"cats" : 0, "dogs" : 1}

def get_images(image_dir):
    """
    returns all image_paths in subfolders and labels them according to their subfolder name.
    """
    picture_files = []

    for root, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, filename)
                last_subfolder = os.path.basename(os.path.dirname(file_path))
                picture_files.append((file_path, LABEL[last_subfolder]))

    labels = [label for _, label in picture_files]
    picture_files = [picture_file for picture_file, _ in picture_files]

    return picture_files ,labels



