from torch.utils.data import Dataset
import cv2
import os
import torch
import pickle


'''
    加载数据集的格式:
	- dataset
		- split
			- *.gw

'''
def getDatasets(dir):
	return os.listdir(dir)

# 将数据分为验证集, 测试集, 训练集
class DewarpingDataset(Dataset):
	def __init__(self, root=r"./dataset", split='train'):	# 这里spilt代表是test, train集, 我不知道该用什么表示
		self.root_dir = root
		self.split_dir = split
		self.path = os.path.join(self.root_dir, self.split_dir)
		self.img_path = os.listdir(self.path)
		self.is_return_img_name = False
		self.row_gap = 1  # value:0, 1, 2;  POINTS NUM: 61, 31, 21
		self.col_gap = 1

	def __len__(self):
		return len(self.img_path)

	def __getitem__(self, index):
		if self.split_dir == 'test':
			img_name = self.img_path[index]
			img_item_path = os.path.join(self.root_dir, self.split_dir, img_name)

			img = cv2.imread(img_item_path, flags=cv2.IMREAD_COLOR)

			img = self.resize_im(img)
			img = self.transform_im(img)

			if self.is_return_img_name:
				return img, img_name
			return img
		else:
			img_name = self.img_path[index]
			img_item_path = os.path.join(self.root_dir, self.split_dir, img_name)

			with open(img_item_path, 'rb') as f:
				perturbed_data = pickle.load(f)

			img = perturbed_data.get('image')
			label = perturbed_data.get('fiducial_points')
			segment = perturbed_data.get('segment')

			img = self.resize_im(img)
			img = img.transpose(2, 0, 1)

			label = self.resize_lbl(label)
			label, segment = self.fiducal_points_lbl(label, segment)
			label = label.transpose(2, 0, 1)

			img = torch.from_numpy(img)
			label = torch.from_numpy(label).float()
			segment = torch.from_numpy(segment).float()

			if self.is_return_img_name:
				return img, label, segment, img_name

			return img, label, segment

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).float()

		return im

	def resize_im(self, im):
		im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
		# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		return im

	def resize_lbl(self, lbl):
		lbl = lbl/[960, 1024]*[992, 992]
		# lbl = lbl/[960, 1024]*[496, 496]
		return lbl

	def fiducal_points_lbl(self, fiducial_points, segment):

		fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
		fiducial_points = fiducial_points[::fiducial_point_gaps[self.row_gap], ::fiducial_point_gaps[self.col_gap], :]
		segment = segment * [fiducial_point_gaps[self.col_gap], fiducial_point_gaps[self.row_gap]]
		return fiducial_points, segment


