import os
import io
import bson
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp
from glob import iglob


bson_file = 'train.bson'
NCORE = 16
max_images = 10000 #7069896

input_dir = os.path.abspath(os.path.join(os.getcwd(), './data'))
base_dir = os.path.join(os.getcwd())
images_dir = os.path.join(base_dir, 'train')
bson_file = os.path.join(input_dir, bson_file)

product_count = 0
category_count = 0
picture_count = 0

def process(q, iolock):
    global product_count
    global category_count
    global picture_count
    while True:
        d = q.get()
        if d is None:
            break

        product_count += 1
        product_id = str(d['_id'])
        category_id = str(d['category_id'])

        category_dir = os.path.join(images_dir, category_id)
        if not os.path.exists(category_dir):
            category_count += 1
            try:
                os.makedirs(category_dir)
            except:
                pass

        for e, pic in enumerate(d['imgs']):
            picture_count += 1
            picture = imread(io.BytesIO(pic['picture']))
            picture_file = os.path.join(category_dir, product_id + '_' + str(e) + '.jpg')
            if not os.path.isfile(picture_file):
                plt.imsave(picture_file, picture)

def split_data(data, ratio):
    random.seed(42)
    random.shuffle(data)
    data_len = len(data)
    split_point = int(data_len * ratio)
    train = data[:split_point]
    val = data[split_point:]

    return list(train), list(val)

q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))


data = bson.decode_file_iter(open(bson_file, 'rb'))

for c, d in tqdm(enumerate(data)):
    if (c + 1) > max_images:
        break
    q.put(d)  # blocks until q below its max size

# tell workers we're done
for _ in range(NCORE):
    q.put(None)
pool.close()
pool.join()

print('Images saved at %s' % images_dir)
print('Products: \t%d\nCategories: \t%d\nPictures: \t%d' % (product_count, category_count, picture_count))

ftrain = open(os.path.join(base_dir, 'train.txt'), 'w')
fval = open(os.path.join(base_dir, 'val.txt'), 'w')
fclass = open(os.path.join(base_dir, 'class.txt'), 'w')

category_list = []
rootdir_glob = images_dir + '/**/*'
fname_list = [f for f in iglob(rootdir_glob)]
train, val = split_data(fname_list, 0.9)
for fname in train:
    category = fname.split('/')[-2]
    ftrain.write(fname + ' ' + category + '\n')
    if category not in category_list:
        category_list.append(category)
        fclass.write(category + '\n')
for fname in val:
    category = fname.split('/')[-2]
    fval.write(fname + ' ' + category + '\n')
    if category not in category_list:
        category_list.append(category)
        fclass.write(category + '\n')


ftrain.close()
fval.close()