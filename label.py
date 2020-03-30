import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import shutil
from pathlib import Path

def press(event):
    sys.stdout.flush()
    Path("celeba/makeup/makeup").mkdir(parents=True, exist_ok=True)
    Path("celeba/no-makeup/no-makeup").mkdir(parents=True, exist_ok=True)
    if event.key == 'left':
        shutil.move(imagePath, "celeba/makeup/makeup/"+str(imagePath.split('/')[1]))
        plt.close()
    elif event.key == 'right':
        shutil.move(imagePath, "celeba/no-makeup/no-makeup/"+str(imagePath.split('/')[1]))
        plt.close()
    elif event.key == 'x':
        Path(imagePath).unlink()
        plt.close()

        
images = glob.glob('img_align_celeba/*.jpg')
for idx, imagePath in enumerate(images):
    print(f'{idx}/{len(images)} {imagePath}')
    image = mpimg.imread(imagePath)
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.imshow(image)
    plt.show()