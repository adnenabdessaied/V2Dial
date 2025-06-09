import os
import re
import json
from tqdm import trange
from utils.dist import is_main_process
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import numpy as np

def open_img(img_pth):
    try:
        img = Image.open(img_pth).convert('RGB')
        return img
    except:
        img = np.random.randint(0, high=256, size=(224,224, 3))
        img = Image.fromarray(img, 'RGB')
    return img


def pre_text(text, max_l=None):
    text = re.sub(r"(['!?\"()*#:;~])", '', text.lower())
    text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    text = re.sub(r"\s{2,}", ' ', text)
    text = text.rstrip('\n').strip(' ')

    if max_l:  # truncate
        words = text.split(' ')
        if len(words) > max_l:
            text = ' '.join(words[:max_l])
    return text


def get_datasets_media(dataloaders):
    media = {}
    for dataloader in dataloaders:
        if isinstance(dataloader.dataset, ConcatDataset):
            media[dataloader.dataset.datasets[0].medium] = dataloader
        else:
            media[dataloader.dataset.medium] = dataloader

    # media = [dataloader.dataset.medium for dataloader in dataloaders]
    return media

def type_transform_helper(x):
    return x.float().div(255.0)

def load_anno(ann_file_list):
    """[summary]

    Args:
        ann_file_list (List[List[str, str]] or List[str, str]):
            the latter will be automatically converted to the former.
            Each sublist contains [anno_path, image_root], (or [anno_path, video_root, 'video'])
            which specifies the data type, video or image

    Returns:
        List(dict): each dict is {
            image: str or List[str],  # image_path,
            caption: str or List[str]  # caption text string
        }
    """
    if isinstance(ann_file_list[0], str):
        ann_file_list = [ann_file_list]

    ann = []
    for d in ann_file_list:
        data_root = d[1]
        fp = d[0]
        is_video = len(d) == 3 and d[2] == "video"
        cur_ann = json.load(open(fp, "r"))
        iterator = trange(len(cur_ann), desc=f"Loading {fp}") \
            if is_main_process() else range(len(cur_ann))
        for idx in iterator:
            key = "video" if is_video else "image"
            video_id = cur_ann[idx][key][5:].split('.')[0]
            # unified to have the same key for data path
            # if isinstance(cur_ann[idx][key], str):
            cur_ann[idx]["vis"] = os.path.join(data_root, video_id)
            # else:  # list
            #     cur_ann[idx]["vis"] = [os.path.join(data_root, e) for e in cur_ann[idx][key]]
        ann += cur_ann
    return ann