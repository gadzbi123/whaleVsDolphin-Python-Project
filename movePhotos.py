import pandas as pd
import shutil as sh
import numpy as np


data = pd.read_csv('./csv/train.csv')
a = data.head(50000)

i = 0
hw_count = 0
bd_count = 0
src_dir = "./happy-whale-and-dolphin/train_images/"

data_to_store = {
    'Img_source': [],
    'Animal': [],
}

for idx, img in enumerate(a.values):
    if img[1] == 'humpback_whale' and hw_count != 250 and idx > 35000:
        dest = "./ds/single_test/" + str(i) + '.jpg'
        sh.copy(src_dir + img[0], dest)
        data_to_store['Img_source'].append(str(i) + '.jpg')
        data_to_store['Animal'].append("humpback_whale")
        hw_count += 1
        i += 1

    if img[1] == 'bottlenose_dolphin' and bd_count != 250 and idx > 35000:
        dest = "./ds/single_test/" + str(i) + '.jpg'
        sh.copy(src_dir + img[0], dest)
        data_to_store['Img_source'].append(str(i) + '.jpg')
        data_to_store['Animal'].append("bottlenose_dolphin")
        bd_count += 1
        i += 1


print("whale " + str(hw_count), ", dolphin " + str(bd_count))
df = pd.DataFrame(data_to_store)
df.to_csv("./csv/test.csv", index=False)
