'''
File: /tf2json.py
Created Date: Friday November 17th 2023
Author: Zihan
-----
Last Modified: Saturday, 18th November 2023 1:51:51 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from PIL import Image
import io
import os
import json
import pandas as pd


def save_images_and_create_json(df, image_col, label_col, base_dir, output_json, img_dir=None):
    img_dir = os.path.join(base_dir, img_dir) if img_dir else base_dir
    os.makedirs(img_dir, exist_ok=True)

    image_label_dict = {}

    for index, row in df.iterrows():
        img_data_dict = row[image_col]
        label = row[label_col]
        img_path = os.path.join(img_dir, f"image_{index}.jpg")

        if img_byte_data := img_data_dict.get('bytes'):
            image_label_dict[img_path] = label
            with Image.open(io.BytesIO(img_byte_data)) as img:
                img.save(img_path)

    with open(os.path.join(base_dir, output_json), 'w') as f:
        json.dump(image_label_dict, f, indent=4)


# 使用函数
baseDir = '/home/zihan/dataset/FGVC/OxfordFlower'
df = pd.read_parquet(
    f'{baseDir}/train-00000-of-00001-12de94e121bdbead.parquet')
# print(df['image'].head())
save_images_and_create_json(
    df, 'image', 'label', baseDir, 'train.json', img_dir='train')

df = pd.read_parquet(f'{baseDir}/test-00000-of-00001-96eeec628415add6.parquet')

# divide test set into val and test
df_val = df.sample(frac=0.5, random_state=42)
df_test = df.drop(df_val.index)

save_images_and_create_json(
    df_val, 'image', 'label', baseDir, 'val.json', img_dir='val')
save_images_and_create_json(
    df_test, 'image', 'label', baseDir, 'test.json', img_dir='test')
