import math
import torch
import torch.nn as nn

from .modules import *

class Root(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, residual):
		super(Root, self).__init__()
		self.conv = nn.Conv1d(
			in_channels, out_channels, kernel_size,
			stride=1, bias=False, padding=(kernel_size - 1) // 2)
		self.bn = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.residual = residual

	def forward(self, *x):
		children = x
		
		x = self.conv(torch.cat(x, 1))
		x = self.bn(x)
		if self.residual:
			x += children[0]
		x = self.relu(x)
		return x

class Tree(nn.Module):
	def __init__(self, levels, block, in_channels, out_channels, stride=1,
				 level_root=False, root_dim=0, root_kernel_size=1,
				 dilation=1, root_residual=False, dsp = True, up_path = True, gate = True):
		super(Tree, self).__init__()
		if root_dim == 0:
			root_dim = 2 * out_channels
		if level_root:
			root_dim += in_channels
		if levels == 1:
			self.tree1 = block(in_channels, out_channels, dilation=dilation, dsp = dsp, up_path = up_path, gate = gate)
			self.tree2 = block(out_channels, out_channels, dilation=dilation, dsp = dsp, up_path = up_path, gate = gate)
		else:
			self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
							  stride, root_dim=0,
							  root_kernel_size=root_kernel_size,
							  dilation=dilation, root_residual=root_residual, dsp = dsp, up_path = up_path, gate = gate)
			self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
							  root_dim=root_dim + out_channels,
							  root_kernel_size=root_kernel_size,
							  dilation=dilation, root_residual=root_residual, dsp = dsp, up_path = up_path, gate = gate)
		if levels == 1:
			self.root = Root(root_dim, out_channels, root_kernel_size,
							 root_residual)
		self.level_root = level_root
		self.root_dim = root_dim
		self.downsample = None
		self.project = None
		self.levels = levels
	
		if in_channels != out_channels:
			self.project = nn.Sequential(
				nn.Conv1d(in_channels, out_channels,
						  kernel_size=1, stride=1, bias=False),
				nn.BatchNorm1d(out_channels)
			)

	def forward(self, x, children=None):
		children = [] if children is None else children
		bottom = x
		if self.level_root:
			children.append(bottom)
		x1 = self.tree1(x)
		if self.levels == 1:
			x2 = self.tree2(x1)
			x = self.root(x2, x1, *children)
		else:
			children.append(x1)
			x = self.tree2(x1, children=children)
		return x


class RawNeXt(nn.Module):
	"""
	RawNeXt model.
	This model is based on the deep layer aggregation (DLA) structure[1].

	Reference:
	[1] Yu, Fisher, et al. 
	"Deep layer aggregation." CVPR. 2018.
	"""
	def __init__(self, levels, channels, code_dim=512, block=Bottleneck, residual_root=False, dsp = True, up_path = True, gate = True, **kwargs):
		super(RawNeXt, self).__init__()

		self.base_layer = nn.Sequential(
			nn.Conv1d(1, channels[0], kernel_size=3, stride=3,
					  padding=0, bias=False),
			nn.BatchNorm1d(channels[0]),
			nn.ReLU(inplace=True))

		self.level0 = self._make_conv_level(
			channels[0], channels[0], levels[0])
		self.level1 = self._make_conv_level(
			channels[0], channels[1], levels[1])
		self.level2 = Tree(levels[2], block, channels[1], channels[2], 1,
						   level_root=False, root_residual=residual_root, dsp = dsp, up_path = up_path, gate = gate)
		self.level3 = Tree(levels[3], block, channels[2], channels[3], 1,
						   level_root=True, root_residual=residual_root, dsp = dsp, up_path = up_path, gate = gate)
		self.level4 = Tree(levels[4], block, channels[3], channels[4], 1,
						   level_root=True, root_residual=residual_root, dsp = dsp, up_path = up_path, gate = gate)
		self.level5 = Tree(levels[5], block, channels[4], channels[5], 1,
						   level_root=True, root_residual=residual_root, dsp = dsp, up_path = up_path, gate = gate)

		
		self.attention = nn.Sequential(
			nn.Conv1d(channels[5], channels[5] // 8, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(channels[5] // 8),
			nn.Conv1d(channels[5] // 8, channels[5], kernel_size=1),
			nn.Softmax(dim=-1),
		)

		self.bn_agg = nn.BatchNorm1d(channels[5] * 2)

		self.fc = nn.Linear(channels[5]*2, code_dim)
		self.bn_code = nn.BatchNorm1d(code_dim)

		self.mp = nn.MaxPool1d(3)

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				n = m.kernel_size[0]  * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
		modules = []
		for i in range(convs):
			modules.extend([
				nn.Conv1d(inplanes, planes, kernel_size=3,
						  stride=stride if i == 0 else 1,
						  padding=dilation, bias=False, dilation=dilation),
				nn.BatchNorm1d(inplanes),
				nn.ReLU(inplace=True)
				])
			inplanes = planes
		return nn.Sequential(*modules)

	def forward(self, x, is_test = False):
		x = self.base_layer(x)

		for i in range(6):
			x = getattr(self, 'level{}'.format(i))(x)
			x = self.mp(x)
		
		w = self.attention(x)
		m = torch.sum(x * w, dim=-1)
		s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		x = torch.cat([m, s], dim=1)
		x =	self.bn_agg(x)

		code = self.fc(x)
		
		code = self.bn_code(code)

		if is_test:	return code
		else:
			code_norm = code.norm(p=2, dim=1, keepdim=True) / 9.0
			code = torch.div(code, code_norm)
			return code
			
def get_RawNeXt(levels, channels, code_dim, dsp, up_path, gate, **kwargs):
	model = RawNeXt(
				levels = levels,
				channels = channels,
				code_dim= code_dim,
				dsp = dsp,
				up_path = up_path,
				gate = gate,
				**kwargs)
	return model