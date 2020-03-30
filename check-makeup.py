import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import shutil
from pathlib import Path

def press(event):
    sys.stdout.flush()
    if event.key == 'left':
        plt.close()
    elif event.key == 'right':
        shutil.move(imagePath, "celeba/no-makeup/no-makeup/"+str(imagePath.split('/')[-1]))
        plt.close()
    elif event.key == 'x':
        Path(imagePath).unlink()
        plt.close()

        
images = glob.glob('celeba/makeup/makeup/*.jpg')
for idx, imagePath in enumerate(reversed(images)):
    print(f'{idx}/{len(images)} {imagePath}')
    image = mpimg.imread(imagePath)
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.imshow(image)
    plt.show()