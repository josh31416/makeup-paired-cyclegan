import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import shutil
from pathlib import Path

class1 = "makeup"
class2 = "no-makeup"
new_dataset_folder = "celeba"
dataset_folder = "img_align_celeba"

def press(event):
    sys.stdout.flush()
    class1_path = os.path.join(new_dataset_folder, class1, class1)
    class2_path = os.path.join(new_dataset_folder, class2, class2)
    
    # create directories if they don't exist
    Path(class1_path).mkdir(parents=True, exist_ok=True)
    Path(class2_path).mkdir(parents=True, exist_ok=True)
    
    # assign to class with keyboard keys
    # left arrow  -> class 1
    # right arrow -> class 2
    # x           -> delete image
    if event.key == 'left':
        shutil.move(imagePath, class1_path+str(imagePath.split('/')[-1]))
        plt.close()
    elif event.key == 'right':
        shutil.move(imagePath, class2_path+str(imagePath.split('/')[-1]))
        plt.close()
    elif event.key == 'x':
        Path(imagePath).unlink()
        plt.close()

        
images = glob.glob(f"{dataset_folder}/*.jpg")
for idx, imagePath in enumerate(images):
    print(f'{idx}/{len(images)} {imagePath}')
    image = mpimg.imread(imagePath)
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.imshow(image)
    plt.show()