from PIL import Image
import os
for d, name in [('/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/low', 'eval15/low'),
                ('/home/user/YouSihai/lightEnhancement/data/Dataset_train/Dataset_train/LOLDataset/our485/low', 'our485/low')]:
    fs = sorted(os.listdir(d))[:8]
    sizes = {}
    for f in fs:
        s = Image.open(os.path.join(d, f)).size
        sizes[f] = s
        print(name, f, s)
    print('---', name, 'total', len(os.listdir(d)))