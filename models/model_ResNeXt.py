import math
import torch
import torch.nn as nn

from .modules import *

class ResNeXt(nn.Module):

	def __init__(
		self, levels, channels, code_dim=512, block = Bottleneck, dsp = False, up_path = False, gate = False, **kwargs
	):  
		super(ResNeXt, self).__init__()
		self.inplanes = channels[0]

		self.base_layer = nn.Sequential(
			nn.Conv1d(1, channels[0], kernel_size=3, stride=3,
					  padding=0, bias=False),
			nn.BatchNorm1d(channels[0]),
			nn.ReLU(inplace=True))
		
		self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
		self.level1 = self._make_conv_level(channels[0], channels[1], levels[1])
		self.level2 = self._make_layer(block, channels[2], levels[2], dsp = dsp, up_path = up_path, gate = gate)
		self.level3 = self._make_layer(block, channels[3], levels[3], dsp = dsp, up_path = up_path, gate = gate)
		self.level4 = self._make_layer(block, channels[4], levels[4], dsp = dsp, up_path = up_path, gate = gate)
		self.level5 = self._make_layer(block, channels[5], levels[5], dsp = dsp, up_path = up_path, gate = gate)

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
			
	def _make_layer(self, block, planes, nb_layer, dsp = False, up_path = False, gate = False):
		layers = []
		for l in range(nb_layer):
			layers.append(block(self.inplanes, planes, dsp = dsp, up_path = up_path, gate = gate))
			self.inplanes = planes
		return nn.Sequential(*layers)

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
	def forward(self, x, is_test=False):
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


def get_ResNeXt(levels, channels, code_dim, dsp, up_path, gate, **kwargs):
	model = ResNeXt(
		levels=levels,
		channels=channels,
		code_dim=code_dim,
		dsp = dsp,
		up_path = up_path,
		gate = gate,
		**kwargs
	)
	return model