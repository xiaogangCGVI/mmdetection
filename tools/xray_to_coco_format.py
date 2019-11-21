import json
import argparse
import re
import os
from PIL import Image
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
from pycocotools.mask import toBbox, area, frPyObjects

def check_chinese(text):
    match = re.findall(r'[\u4e00-\u9fff]+', text)
    return len(match) > 0

train_label_file = '/home/yilxiong/xray-detection/class_idx.txt'
test_label_file = '/home/yilxiong/xray-detection/test_idx.txt'

def parse_args():
    parser = argparse.ArgumentParser(description = 'Convert annotation format from xray to coco')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--data_folder', '-d', default = None)
    return parser.parse_args()
    
def main():
    args = parse_args()
    with open(train_label_file) as f:
        label = [i.strip().split('\t') for i in f.readlines()]
    label2id = {k: int(v) for k, v in label}
    with open(args.input) as f:
        data = [json.loads(i)['data']['img'] for i in f.readlines()]
    r = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': label_id, 'name': label} for label, label_id in label2id.items() if not check_chinese(label)]
    }
    img_id = 0
    seg_id = 0
    for annotation in data:
        img_name = annotation['location']
        print(img_name)
        if 'invalid' in annotation and annotation['invalid']:
            print("Invalid image: {}".format(img_name))
            continue
        try: 
            im = Image.open(os.path.join(args.data_folder, img_name))
            width, height = im.size
        except:
            print("Fail to open image: {}".format(img_name))
            continue
        img_id += 1
        r['images'].append({'file_name': img_name, 'id': img_id,
                            'height': height, 'width': width})
        segmentations = annotation['polygons']
        for seg in segmentations:
            label_name = seg['tag']
            if label_name not in label2id:
                print("Label not found: {}".format(label_name))
                label_name = 'others'
            seg_id += 1
            label_id = label2id[label_name]
            mask = []
            for pnt in seg['points']:
                mask += [pnt['x']*width, pnt['y']*height]
            mask = [mask]
            rle = frPyObjects(mask, height, width)[0]
            annotation = {
                'category_id': label_id,
                'id': seg_id,
                'image_id': img_id,
                'iscrowd': 0,
                'bbox': toBbox(rle).tolist(),
                'area': float(area(rle)),
                'segmentation': mask
            }
            r['annotations'].append(annotation)
    with open(args.output, 'w') as f:
        json.dump(r, f)

if __name__ == "__main__":
    main()
