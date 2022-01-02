import os
import glob

from jittor.dataset.dataset import Dataset
import jittor as jt
from PIL import Image
import numpy as np
from torch import max_pool1d
import copy


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100,batch_size=16,
                 shuffle=False,
                 num_workers=0):
        super().__init__()
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_len = len(self.paths)
        #self.num_workers=num_workers
        self._disable_workers=False
        self.set_attrs(batch_size = batch_size,shuffle = shuffle,total_len=self.total_len)

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = jt.float32(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text if c in self.CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)
            
            return image, target, target_length
        else:
            return image

    def collate_batch(self, batch):

        if self.texts is None:
            images=batch
            images = jt.stack(images, dim=0) 
            #print(images.shape)   
            return images


        images, targets, target_lengths = zip(*batch)
        images = jt.stack(images, dim=0)
        
        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.numpy().max()
        targets = [target.reindex([max_target_length.item()], ["i0"]) for target in targets]
        targets = jt.stack(targets, dim=0)


        return images, targets, target_lengths
    
    
class SVTDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100,batch_size=16,
                 shuffle=False,
                 num_workers=0):
        super().__init__()
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_len = len(self.paths)
        #self.num_workers=num_workers
        self._disable_workers=False
        self.set_attrs(batch_size = batch_size,shuffle = shuffle,total_len=self.total_len)

    def _load_from_raw_files(self, root_dir, mode):
       
       
        paths_file='SVT_test_647.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, text = line.strip().split(' ')
                text=text.lower()
                path = os.path.join(root_dir+'images/test/', path)
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = jt.float32(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text if c in self.CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)
            
            return image, target, target_length
        else:
            return image

    def collate_batch(self, batch):

        if self.texts is None:
            images=batch
            images = jt.stack(images, dim=0) 
            #print(images.shape)   
            return images

        images, targets, target_lengths = zip(*batch)
        images = jt.stack(images, dim=0)
        
        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.numpy().max()
        targets = [target.reindex([max_target_length.item()], ["i0"]) for target in targets]
        targets = jt.stack(targets, dim=0)


        return images, targets, target_lengths



    
class IIITDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100,batch_size=16,
                 shuffle=False,
                 num_workers=0):
        super().__init__()
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_len = len(self.paths)
        #self.num_workers=num_workers
        self._disable_workers=False
        self.set_attrs(batch_size = batch_size,shuffle = shuffle,total_len=self.total_len)

    def _load_from_raw_files(self, root_dir, mode):
       
       
        paths_file= 'label/IIIT_test_3000.txt'
        # paths_file= 'label/ICDAR2003_test_1110.txt'
        
        # paths_file= 'label/ICDAR2013_test_1095.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                l = line.strip().split(' ')
                path = l[0]
                text = ""
                for i in range(1, len(l)):
                    text += l[i]
                text=text.lower()
                path = os.path.join(root_dir+'images/test/', path)
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = jt.float32(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text if c in self.CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)
            
            return image, target, target_length
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths = zip(*batch)
        images = jt.stack(images, 0)
        targets = jt.cat(targets, 0)
        target_lengths = jt.cat(target_lengths, 0)
        return images, targets, target_lengths



