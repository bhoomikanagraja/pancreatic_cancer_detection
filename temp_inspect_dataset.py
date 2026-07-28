from PIL import Image
import os

for path in [
    'archive/Dataset/X_train/1.png',
    'archive/Dataset/Y_train/1.png',
    'archive/Dataset/X_test/1.png',
    'archive/Dataset/Y_test/1.png',
]:
    img = Image.open(path)
    print(path, img.mode, img.size)
    pixels = list(img.getdata())
    print('unique sample', sorted(set(pixels))[:20], '... total', len(set(pixels)))
    print('first 10', pixels[:10])
    print()
