import os
import torch
import cv2
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, ConcatDataset

kitti_dataset = {'00': [0, 4540],
                 '01': [0, 1100],
                 '02': [0, 4660],
                 '03': [0, 800],
                 '04': [0, 270],
                 '05': [0, 2760],
                 '06': [0, 1100],
                 '07': [0, 1100],
                 '08': [1100, 5170],
                 '09': [0, 1590],
                 '10': [0, 1200]}

# Generate the data frames per path
def get_data_info(dataset_dir, kitti_sequence, seq_len=5, overlap=1, drop_last=True):
    image_dir = os.path.join(dataset_dir, 'images')
    pose_dir = os.path.join(dataset_dir, 'poses/local_poses')

    img_sequence, pose_sequence= [], []

    # Load & sort the raw data
    poses = np.loadtxt(os.path.join(pose_dir, f'{kitti_sequence}.txt'))  # (n_images, 6)
    images = natsorted([f for f in os.listdir(os.path.join(image_dir, kitti_sequence, 'image_2')) if f.endswith('.png')])
    n_frames = len(images)
    start = 0
    
    while (start + seq_len) < n_frames:
        x_seg = images[start:start+seq_len]
        img_sequence.append([os.path.join(image_dir, kitti_sequence, 'image_2', img) for img in x_seg])
        pose_sequence.append(poses[start:start+seq_len-1])
        start += seq_len - overlap

    if not drop_last:
        img_sequence.append([os.path.join(image_dir, kitti_sequence, img) for img in images[start:]])
        pose_sequence.append(poses[start:])

    # Store in a dictionary
    data = {'image_path': img_sequence, 'pose':pose_sequence}
    return data

def load_dataset(dataset_dir, batch_size=2, seq_len=5, width=640, height=192, shuffle=True):
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for seq, _ in kitti_dataset.items():
        dataset = ImageSequenceDataset(dataset_dir=dataset_dir, kitti_sequence=seq, seq_len=seq_len, width=width, height=height)

        if seq == '04' or seq=='07':
            valid_datasets.append(dataset)

        elif seq == '09' or seq == '10':
            test_datasets.append(dataset)

        else:
            train_datasets.append(dataset)

    train_loader = DataLoader(dataset=ConcatDataset(train_datasets),
                              batch_size=batch_size,
                              shuffle=shuffle)

    valid_loader = DataLoader(dataset=ConcatDataset(valid_datasets),
                             batch_size=batch_size,
                             shuffle=shuffle)
    
    test_loader = DataLoader(dataset=ConcatDataset(test_datasets),
                             batch_size=batch_size,
                             shuffle=shuffle)

    return train_loader, valid_loader, test_loader

class ImageSequenceDataset(Dataset):
    def __init__(self, dataset_dir, kitti_sequence, seq_len=5, width=640, height=192, overlap=1, drop_last=True):
        self.data_info = get_data_info(dataset_dir, kitti_sequence, seq_len, overlap, drop_last)
        self.image = self.data_info['image_path']
        self.pose = self.data_info['pose']
        self.width = width
        self.height = height

    def transform(self, img):
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        tensor_img = tensor_img.unsqueeze(0) # [B, C, H, W]

        return tensor_img

    def __len__(self):
        return len(self.data_info['image_path'])

    def __getitem__(self, index):
        image_sequence = []
        image_path_sequence = self.image[index]

        for img_path in image_path_sequence:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            image_sequence.append(img)

        image_sequence = torch.cat(image_sequence, 0) # torch.Size([7, 3, 192, 640]), which is (sequence length, channel, height, width)
        pose_sequence = torch.tensor(self.pose[index], dtype=torch.float32)

        return image_sequence, pose_sequence

if __name__=='__main__':
    dataset_dir = '/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/Dataset/'
    train_loader, valid_loader, test_loader = load_dataset(dataset_dir, batch_size=8, shuffle=True)

    for img, pose in train_loader:
        print(img.size(), pose.size())