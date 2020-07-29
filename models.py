import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import gl

# DEBUG SWITCH
DEBUG_MODEL = False
# is the internet connection?
IS_NET_CONNECTION = True


class GraphConvolution(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""
	
	def __init__(self, in_features, out_features, bias=False):
		
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.Tensor(1, 1, out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		
		self.Linear_predict = nn.Linear(1000, 3000)
	
	def reset_parameters(self):
		
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
	
	def forward(self, input, adj):
		support = torch.matmul(input, self.weight)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output
	
	def __repr__(self):
		
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'


class GCNResnet(nn.Module):
	def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None):
		super(GCNResnet, self).__init__()
		self.state = {}
		self.state['use_gpu'] = torch.cuda.is_available()
		self.opt = option
		self.is_usemfb = option.IS_USE_MFB
		self.pooling_stride = option.pooling_stride
		self.num_classes = num_classes
		self.gamma_weight = option.gamma_weight
		self.features = nn.Sequential(
			model.conv1,
			model.bn1,
			model.relu,
			model.maxpool,
			model.layer1,
			model.layer2,
			model.layer3,
			model.layer4,
		)
		attetion_out_channel_1 = 1024
		attetion_out_channel_2 = 1024
		self.attentionNet = nn.Sequential(
			nn.Conv2d(2048, attetion_out_channel_1, kernel_size=1, padding=1),
			nn.BatchNorm2d(attetion_out_channel_1),
			nn.ReLU(inplace=True),
			nn.Conv2d(attetion_out_channel_1, attetion_out_channel_2, kernel_size=3),
			nn.BatchNorm2d(attetion_out_channel_2),
			nn.ReLU(inplace=True),
			nn.Conv2d(attetion_out_channel_2, self.num_classes, kernel_size=1),
			nn.BatchNorm2d(self.num_classes),
			nn.ReLU(inplace=True),
			nn.Softmax2d(),
		)
		linear_out_channel_1 = 1024
		self.linearNet = nn.Sequential(
			nn.Conv2d(2048, linear_out_channel_1, kernel_size=1),
			nn.BatchNorm2d(linear_out_channel_1),
			nn.ReLU(inplace=True),
			nn.Conv2d(linear_out_channel_1, self.num_classes, kernel_size=1),
			nn.Sigmoid(),
		)
		link_channel_1 = 1024
		link_channel_2 = 2048
		self.linkNet = nn.Sequential(
			nn.Conv2d(self.num_classes, link_channel_1, kernel_size=1, padding=1),
			nn.Conv2d(link_channel_1, linear_out_channel_1, kernel_size=3),
			nn.Conv2d(link_channel_1, link_channel_2, kernel_size=1),
		)
		
		self.pooling = nn.MaxPool2d(14, 14)
		
		self.gc1 = GraphConvolution(in_channel, 1024)
		self.gc2 = GraphConvolution(1024, 2048)
		self.relu = nn.LeakyReLU(0.2)
		
		_adj = gen_A(self.opt.threshold_p, num_classes, self.opt.threshold_tao, adj_file)
		self.A = Parameter(torch.from_numpy(_adj).float())
		
		self.image_normalization_mean = [0.485, 0.456, 0.406] 
		self.image_normalization_std = [0.229, 0.224, 0.225]  
		
		self.JOINT_EMB_SIZE = option.linear_intermediate
		assert self.JOINT_EMB_SIZE%self.pooling_stride==0, \
			'linear-intermediate value must can be divided exactly by sum pooling stride value!'
		
		if self.is_usemfb:
			assert self.JOINT_EMB_SIZE % self.pooling_stride == 0, \
				'linear-intermediate value must can be divided exactly by sum pooling stride value!'
			self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
			self.ML_fc_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), int(self.num_classes))
		else:
			self.out_in_tmp = int(1)

		self.Linear_imgdataproj = nn.Linear(option.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)  
		self.Linear_classifierproj = nn.Linear(option.CLASSIFIER_CHANNEL,self.JOINT_EMB_SIZE)

	
	def forward(self, feature, inp):
		feature = self.features(feature)
		
		attention_feature = self.attentionNet(feature)
		linear_feature = self.linearNet(feature)
		link_features = self.linkNet(torch.mul(attention_feature, linear_feature))
		new_feature = self.pooling(torch.add(self.gamma_weight*feature,
		                                 (1-self.gamma_weight)*link_features))
		feature = new_feature.view(new_feature.size(0), -1) 
		inp = inp[0]
		adj = gen_adj(self.A).detach()
		x = self.gc1(inp, adj)
		x = self.relu(x)
		x = self.gc2(x, adj)  
		x = th.transpose(x, 0, 1)
		if self.is_usemfb:
			if self.state['use_gpu']:
				x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
			else:
				x_out = torch.FloatTensor(torch.FloatStorage())
			for i_row in range(int(feature.shape[0])):
				img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
				if self.state['use_gpu']:
					out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
				else:
					out_row = torch.FloatTensor(torch.FloatStorage())
				for col in range(int(x.shape[1])):  
					tmp_x = x[:, col].view(1, -1) 
					classifier_linear = self.Linear_classifierproj(tmp_x)  
					iq = torch.mul(img_linear_row, classifier_linear)  
					iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)  
					iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2)  
					
					if self.out_in_tmp != 1:  
						temp_out = self.ML_fc_layer(out_row)
						out_row = temp_out  
					
					out_row = torch.cat((out_row,iq),1)
					
				if self.out_in_tmp!=1: 
					temp_out = self.ML_fc_layer(out_row)
					out_row = temp_out          
				x_out = torch.cat((x_out, out_row),0)
		else:
			x_out = th.matmul(feature, x)
		
		return x_out
	
	def get_config_optim(self, lr, lrp):
		return [
			{'params': self.features.parameters(), 'lr': lr * lrp},
			{'params': self.attentionNet.parameters(), 'lr': lr},
			{'params': self.linearNet.parameters(), 'lr': lr},
			{'params': self.linkNet.parameters(), 'lr': lr},
			{'params': self.gc1.parameters(), 'lr': lr},
			{'params': self.gc2.parameters(), 'lr': lr},
		]
	
	@property
	def display_model_hyperparameters(self):
		print("self.is_usetanh = ",self.is_usetanh)


def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, in_channel=300):
	'''
	:param opt:         from config.py
	:param num_classes: the amount of num_classes
	:param t:           this value corresponding with the "threshold tao" when construct the correlation matrix
	:param pretrained:  use pretrained resnet101 or not
	:param adj_file:    /data/voc/voc_adj.pkl file or /data/coco/coco_adj.pkl file
	:param in_channel:  input dimensionality of the word-to-vector size
	:return:
	'''
	if IS_NET_CONNECTION:
		model = models.resnet101(pretrained=pretrained)
	else:
		model = models.resnet101(pretrained=False)
		model.load_state_dict(torch.load('./checkpoint/pretrained_resnet101/resnet101-5d3b4d8f.pth'))
	
	return GCNResnet(opt, model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
