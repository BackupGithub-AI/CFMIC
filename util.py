import math, os, sys
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch.nn.functional as F
from cauchy_hash import *

# DEBUG switch
DEBUG_UTIL = False


class Warp(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		# if DEBUG_UTIL:
		#     #print("FILE:util.py CLASS:Warp INIT\n")
		
		self.size = int(size)
		self.interpolation = interpolation
	
	def __call__(self, img):
		# if DEBUG_UTIL:
		#     #print("FILE:util.py CLASS:Warp FUNC:__call__\n")
		
		return img.resize((self.size, self.size), self.interpolation)
	
	def __str__(self):
		# if DEBUG_UTIL:
		#     #print("FILE:util.py CLASS:Warp FUNC:__str__\n")
		return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
		                                                                                        interpolation=self.interpolation)


class MultiScaleCrop(object):
	'''
	Get many images which have different scale
	'''
	
	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
		
		self.scales = scales if scales is not None else [1, 875, .75, .66]
		self.max_distort = max_distort
		self.fix_crop = fix_crop
		self.more_fix_crop = more_fix_crop
		self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
		self.interpolation = Image.BILINEAR  # bilinear interpolation (双线性插值)
	
	def __call__(self, img):
		
		im_size = img.size
		crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
		crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
		ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
		return ret_img_group
	
	def _sample_crop_size(self, im_size):
		image_w, image_h = im_size[0], im_size[1]
		
		# find a crop size
		base_size = min(image_w, image_h)
		crop_sizes = [int(base_size * x) for x in self.scales]          # data augmentation?
		crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
		crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
		
		pairs = []
		for i, h in enumerate(crop_h):
			for j, w in enumerate(crop_w):
				if abs(i - j) <= self.max_distort:
					pairs.append((w, h))
		
		crop_pair = random.choice(pairs)
		if not self.fix_crop:
			w_offset = random.randint(0, image_w - crop_pair[0])
			h_offset = random.randint(0, image_h - crop_pair[1])
		else:
			w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
		
		return crop_pair[0], crop_pair[1], w_offset, h_offset
	
	def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
		offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
		return random.choice(offsets)
	
	@staticmethod
	def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
		
		w_step = (image_w - crop_w) // 4
		h_step = (image_h - crop_h) // 4
		
		ret = list()
		ret.append((0, 0))  # upper left
		ret.append((4 * w_step, 0))  # upper right
		ret.append((0, 4 * h_step))  # lower left
		ret.append((4 * w_step, 4 * h_step))  # lower right
		ret.append((2 * w_step, 2 * h_step))  # center
		
		if more_fix_crop:
			ret.append((0, 2 * h_step))  # center left
			ret.append((4 * w_step, 2 * h_step))  # center right
			ret.append((2 * w_step, 4 * h_step))  # lower center
			ret.append((2 * w_step, 0 * h_step))  # upper center
			
			ret.append((1 * w_step, 1 * h_step))  # upper left quarter
			ret.append((3 * w_step, 1 * h_step))  # upper right quarter
			ret.append((1 * w_step, 3 * h_step))  # lower left quarter
			ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
		
		return ret
	
	def __str__(self):
		# if DEBUG_UTIL:
		#     #print("FILE:util.py CLASS:MultiScaleCrop FUNC:__str__")
		return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
	"""Download a URL to a local file.

	Parameters
	----------
	url : str
		The URL to download.
	destination : str, None
		The destination of the file. If None is given the file is saved to a temporary directory.
	progress_bar : bool
		Whether to show a command-line progress bar while downloading.

	Returns
	-------
	filename : str
		The location of the downloaded file.

	Notes
	-----
	Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
	"""
	
	def my_hook(t):
		last_b = [0]
		
		def inner(b=1, bsize=1, tsize=None):
			if tsize is not None:
				t.total = tsize
			if b > 0:
				t.update((b - last_b[0]) * bsize)
			last_b[0] = b
		
		return inner
	
	if progress_bar:
		with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
			filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
	else:
		filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
	"""
	The APMeter measures the average precision per class.
	The APMeter is designed to operate on `NxK` Tensors `output` and
	`target`, and optionally a `Nx1` Tensor weight where (1) the `output`
	contains model output scores for `N` examples and `K` classes that ought to
	be higher when the model is more convinced that the example should be
	positively labeled, and smaller when the model believes the example should
	be negatively labeled (for instance, the output of a sigmoid function); (2)
	the `target` contains only values 0 (for negative examples) and 1
	(for positive examples); and (3) the `weight` ( > 0) represents weight for
	each sample.
	"""
	
	def __init__(self, difficult_examples=False):
		super(AveragePrecisionMeter, self).__init__()
		self.reset()
		self.difficult_examples = difficult_examples
		# #print("Class AveragePrecisionMeter initiates over...")
	
	def reset(self):
		"""Resets the meter with empty member variables"""
		self.scores = torch.FloatTensor(torch.FloatStorage())  # self.scores will store all the dataset(train_set or test set) output info
		# #print("In the class AveragePrecisionMeter function reset(): self.score.shape=,self.score=,self.score.type",
		#       self.scores.shape, "\n",self.scores,"\n",type(self.scores))
		self.targets = torch.LongTensor(torch.LongStorage())  # self.scores will store all the dataset(train_set or test set) labels info
	
	def add(self, output, target):
		"""
		Args:
			output (Tensor): NxK tensor that for each of the N examples
				indicates the probability of the example belonging to each of
				the K classes, according to the model. The probabilities should
				sum to one over all classes
			target (Tensor): binary NxK tensor that encodes which of the K
				classes are associated with the N-th input
					(eg: a row [0, 1, 0, 1] indicates that the example is
						 associated with classes 2 and 4)
			weight (optional, Tensor): Nx1 tensor representing the weight for
				each example (each weight > 0)
		"""
		# transform the type of `output` and `target` into tensor
		if not torch.is_tensor(output):
			output = torch.from_numpy(output)
		if not torch.is_tensor(target):
			target = torch.from_numpy(target)
		
		if output.dim() == 1:
			output = output.view(-1, 1)  # transform the tensor into one column
		else:
			assert output.dim() == 2, \
				'wrong output size (should be 1D or 2D with one column \
				per class)'
		if target.dim() == 1:
			target = target.view(-1, 1)
		else:
			assert target.dim() == 2, \
				'wrong target size (should be 1D or 2D with one column \
				per class)'
		if self.scores.numel() > 0:  # Returns the total number of elements in the input tensor.
			assert target.size(1) == self.targets.size(1), \
				'dimensions for output should match previously added examples.'
		
		# make sure storage is of sufficient size
		if self.scores.storage().size() < self.scores.numel() + output.numel():  # tensor.storage() -> Returns the underlying storage
			new_size = math.ceil(self.scores.storage().size() * 1.5)
			self.scores.storage().resize_(int(new_size + output.numel()))
			self.targets.storage().resize_(int(new_size + output.numel()))
		
		# store scores and targets
		offset = self.scores.size(0) if self.scores.dim() > 0 else 0
		# resize_ -> https://pytorch.org/docs/stable/tensors.html?highlight=resize_#torch.Tensor.resize_
		self.scores.resize_(offset + output.size(0), output.size(1))
		self.targets.resize_(offset + target.size(0), target.size(1))
		# narrow-https://pytorch.org/docs/stable/torch.html?highlight=narrow#torch.narrow
		self.scores.narrow(0, offset, output.size(0)).copy_(output)
		self.targets.narrow(0, offset, target.size(0)).copy_(target)
		
		# output -> size:[batchsize, num_classes]
		# target -> size:[batchsize, num_classes] ,includes the ground truth labels info
		#print("AveragePrecisionMeter add output=\n{0},\n{1},\n{2}".format(output, type(output), output.shape))
		#print("AveragePrecisionMeter add target=\n{0},\n{1},\n{2}".format(target, type(target), target.shape))
	
	def value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		if self.scores.numel() == 0:
			return 0
		# in terms of the hash task, self.scores.size() = [test_set_sample_num, Hash bit(here is 64)]
		# self.scores includes all test set samples via the model generate output vectors , voc2007 is 4592 * 64
		#print("\nvalue func: self.scores=\n{0},\n{1},\n{2}".format(type(self.scores), self.scores.size(), self.scores))
		ap = torch.zeros(self.scores.size(1))
		rg = torch.arange(1, self.scores.size(0)).float()
		# compute average precision for each class
		for k in range(self.scores.size(1)):  # k from 0 to num_classes
			#print("\nin for loop: k=", k, "\n")
			# sort scores
			# k from 0 to num_classes-1
			scores = self.scores[:, k]  # get vector according the column
			targets = self.targets[:, k]  # get vector according the column
			# compute average precision, call the static method use the class name
			#print("on value func for loop:\n")
			#print("scores=\n{0},\n{1},\n{2}".format(type(scores), scores.shape, scores))
			#print("targets=\n{0},\n{1},\n{2}".format(type(targets), targets.shape, targets))
			ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
			#print("next loop step\n")
		return ap
	
	@staticmethod
	def average_precision(output, target, difficult_examples=True):
		# print("AveragePrecisionMeter average_precision output=\n{0},\n{1},\n{2}". \
		#       format(type(output), output.shape, output))
		# print("AveragePrecisionMeter average_precision target=\n{0},\n{1},\n{2}". \
		#       format(type(target), target.shape, target))
		# sort examples
		sorted, indices = torch.sort(output, dim=0, descending=True)
		# #print("(util-6)util.py file average_precision func:\n","type(sorted) = ",type(sorted),
		#       "\nsorted.shape = ",sorted.shape,"\ntype(indices) = ",type(indices),"\nindices.shape = ",indices.shape)
		
		# Computes prec@i
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		# #print("(util-7)util.py file average_precision func: type(indices), indices",type(indices),'\n', indices,"\n")
		
		for i in indices:
			# #print("(util-8)util.py file average_precision func: i, type(i)",i,type(i),"\n")
			label = target[i]
			# #print("(util-9)util.py file average_precision func: label, type(label)", label, type(label), "\n")
			if difficult_examples and label == 0:
				continue
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i
	
	def overall(self):
		if self.scores.numel() == 0:
			return 0
		scores = self.scores.cpu().numpy()
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		return self.evaluation(scores, targets)
	
	def overall_topk(self, k):
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		n, c = self.scores.size()
		scores = np.zeros((n, c)) - 1
		index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
		tmp = self.scores.cpu().numpy()
		for i in range(n):
			for ind in index[i]:
				scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
		return self.evaluation(scores, targets)
	
	def evaluation(self, scores_, targets_):
		# print("AveragePrecisionMeter evaluation scores_=\n{0},\n{1},\n{2}". \
		#       format(type(scores_), scores_.shape, scores_))
		# print("AveragePrecisionMeter evaluation targets_=\n{0},\n{1},\n{2}". \
		#       format(type(targets_), targets_.shape, targets_))
		
		n, n_class = scores_.shape
		Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
		for k in range(n_class):
			scores = scores_[:, k]
			targets = targets_[:, k]
			targets[targets == -1] = 0
			Ng[k] = np.sum(targets == 1)
			Np[k] = np.sum(scores >= 0)
			Nc[k] = np.sum(targets * (scores >= 0))
		Np[Np == 0] = 1
		OP = np.sum(Nc) / np.sum(Np)
		OR = np.sum(Nc) / np.sum(Ng)
		OF1 = (2 * OP * OR) / (OP + OR)
		
		CP = np.sum(Nc / Np) / n_class
		CR = np.sum(Nc / Ng) / n_class
		CF1 = (2 * CP * CR) / (CP + CR)
		return OP, OR, OF1, CP, CR, CF1


class HashAveragePrecisionMeter(AveragePrecisionMeter):
	def __init__(self, difficult_examples=False):
		AveragePrecisionMeter.__init__(self, difficult_examples)
	
	def loss_value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		if self.scores.numel() == 0:
			return 0
		target_gt = self.targets
		all_output = self.scores
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		epoch_loss = calcloss.forward(target_gt, all_output)
		
		return epoch_loss
	
	def batch_loss_value(self, batch_target, batch_output):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		target_gt = batch_target
		all_output = batch_output
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		batch_loss = calcloss.forward(target_gt, all_output)
		
		# print("HashAveragePrecisionMeter batch_loss_value:batch_loss=",batch_loss)
		# sys.exit()
		return batch_loss


def gen_A(p, num_classes, t, adj_file):
	'''
	generate the adjecent matrix
	:param opt: get command parameters
	:param num_classes: the amount of classes
	:param t:
	:param adj_file:    word embeding matrix???
	:return:
	'''
	import pickle
	#print("util.py says:t={0},p={1}".format(t, p))
	with open(adj_file, "rb") as f:
		result = pickle.load(f)
	_adj = result['adj']
	_nums = result['nums']
	_nums = _nums[:, np.newaxis]  # increase a dimention
	_adj = _adj / _nums
	_adj[_adj < t] = 0  # this t is the threshold 'tao' in the formula (7)
	_adj[_adj >= t] = 1
	# _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)  # 0.25 denotes the p value in the formula (8)
	# re-weighted scheme of A
	_adj = _adj * p / (_adj.sum(0, keepdims=True) + 1e-6)
	_adj = _adj + np.identity(num_classes, np.int)
	return _adj


def gen_adj(A):
	'''
	:param A:
	:return:
	'''
	D = torch.pow(A.sum(1).float(), -0.5)  # sum element according to the dimension 1
	D = torch.diag(D)
	adj = torch.matmul(torch.matmul(A, D).t(), D)  # (A*D)_T * D = D_T * A * D
	return adj

#
# if __name__ == "__main__":
#     # voc_adj.pkl path
#     dir_voc_adj = "./data/voc/voc_adj.pkl"
#     y = gen_A(20, 0.4, str(dir_voc_adj))
#     #print(y)
