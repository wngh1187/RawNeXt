import torch
import torch.nn as nn

class Gate_module(nn.Module):
	def __init__(self, channels, bottleneck=128, nb_input=3):
		super(Gate_module, self).__init__()
		self.nb_input = nb_input
		self.aap = nn.AdaptiveAvgPool1d(1)
		self.attention = nn.Sequential(
			nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.BatchNorm1d(bottleneck),
			nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
			nn.Softmax(dim = -1),
			)

	def forward(self, input):	
		x = self.aap(input).reshape(input.size(0),-1,self.nb_input)
		x = self.attention(x)
		
		output = None
		for i in range(self.nb_input):
			aw = x[:,:,i].unsqueeze(-1)
			if output == None: output = aw * input[:, x.size(1) * i : x.size(1) * (i+1)]
			else: output += aw * input[:, x.size(1) * i : x.size(1) * (i+1)]
		return output

class Bottleneck(nn.Module):
	"""
	Bottleneck block of ResNeXt architectures[1].
	Dynamic scaling policy (DSP) is based on the elastic module[2].

	Reference:
	[1] Xie, Saining, et al. 
	"Aggregated residual transformations for deep neural networks." CVPR. 2017.
	[2] Wang, Huiyu, et al. 
	"Elastic: Improving cnns with dynamic scaling policies." CVPR. 2019.
	"""
	
	cardinality = 32

	def __init__(self, inplanes, planes, dsp, up_path, gate, stride=1, dilation=1):
		super(Bottleneck, self).__init__()
		
		self.dsp = dsp
		self.up_path = up_path
		self.gate = gate
		cardinality = Bottleneck.cardinality
		bottel_plane = planes
		
		if self.dsp:			
			cardinality = cardinality // 2
			bottel_plane = bottel_plane // 2
			cardinality_split = cardinality
			bottel_plane_split = bottel_plane

			if self.up_path:
				cardinality_split = cardinality_split // 2
				bottel_plane_split = bottel_plane_split // 2
				
		# if change in number of filters
		if inplanes != planes: 
			self.shortcut = nn.Sequential(
				nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
			)

		# original resolution path (original branches)
		self.conv1 = nn.Conv1d(inplanes, bottel_plane,
							   kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(bottel_plane)
		self.conv2 = nn.Conv1d(bottel_plane, bottel_plane, kernel_size=3,
							   stride=stride, padding=dilation, bias=False,
							   dilation=dilation, groups=cardinality)
		self.bn2 = nn.BatchNorm1d(bottel_plane)
		self.conv3 = nn.Conv1d(bottel_plane, planes,
							   kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm1d(planes)
		self.relu = nn.ReLU(inplace=True)

		if self.dsp:
			# low resolution path (down-sampling + up-sampling branches)
			self.pool = nn.AvgPool1d(3, stride=3)
				
			self.conv1_d= nn.Conv1d(
			inplanes, bottel_plane_split, kernel_size=1, bias=False)
			self.bn1_d = nn.BatchNorm1d(bottel_plane_split)

			self.conv2_d = nn.Conv1d(
				bottel_plane_split, bottel_plane_split, kernel_size=3, stride=1, padding=1, bias=False, groups=cardinality_split
			)
			self.bn2_d = nn.BatchNorm1d(bottel_plane_split)

			self.conv3_d= nn.Conv1d(
			bottel_plane_split, planes, kernel_size=1, bias=False)

			self.conv_t = nn.ConvTranspose1d(
				planes, planes, kernel_size =3, stride=3, padding=0)

			if self.up_path:
				# high resolution path (up-sampling + down-sampling branches)
				self.conv_t_u = nn.ConvTranspose1d(
				inplanes, inplanes, kernel_size =3, stride=3, padding=0)

				self.conv1_u= nn.Conv1d(
				inplanes, bottel_plane_split, kernel_size=1, bias=False)
				self.bn1_u = nn.BatchNorm1d(bottel_plane_split)

				self.conv2_u = nn.Conv1d(
					bottel_plane_split, bottel_plane_split, kernel_size=3, stride=1, padding=1, bias=False, groups=cardinality_split
				)
				self.bn2_u = nn.BatchNorm1d(bottel_plane_split)

				self.conv3_u= nn.Conv1d(
				bottel_plane_split, planes, kernel_size=1, bias=False)
				
				# Multi-head attention-based gate module
				if self.gate: self.gate_moduel = Gate_module(planes, planes //3, nb_input = 3)
			else: 
				if self.gate: self.gate_moduel = Gate_module(planes, planes //2, nb_input = 2)
		
		
	def forward(self, x, residual=None):
		if residual is None:
			residual = self.shortcut(x) if hasattr(self, "shortcut") else x
		
		out = self.conv1(x)
		out = self.conv2(self.relu(self.bn1(out)))
		out = self.conv3(self.relu(self.bn2(out)))
		
		if self.dsp:
			# down-sampling
			x_d = self.pool(x)

			out_d = self.conv1_d(x_d)
			out_d = self.conv2_d(self.relu(self.bn1_d(out_d)))    
			out_d = self.conv3_d(self.relu(self.bn2_d(out_d)))

			# up-sampling
			out_d = self.conv_t(out_d)

			if self.up_path:
				# up-sampling
				x_u = self.conv_t_u(x)

				out_u = self.conv1_u(x_u)
				out_u = self.conv2_u(self.relu(self.bn1_u(out_u)))    
				out_u = self.conv3_u(self.relu(self.bn2_u(out_u)))

				# down-sampling
				out_u = self.pool(out_u)
				
				# agregation of features using gate module
				if self.gate: 
					out_cat = torch.cat((out, out_d, out_u), 1)
					out = self.gate_moduel(out_cat)
				# agregation of features using element-wise summation
				else: out += out_d + out_u

			else:
				if self.gate: 
					out_cat = torch.cat((out, out_d), 1)
					out = self.gate_moduel(out_cat)
				else: out += out_d
		
		out = self.bn3(out)

		out += residual
		out = self.relu(out)

		return out
