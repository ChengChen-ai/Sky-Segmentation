import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

tensor2image = lambda T : (127.5*(T[0].cpu().float().numpy()+ 1.0)).astype(np.uint8)

def tensor2im(input_imagae, imtype = np.uint8):
	if not isinstance(input_imagae,imtype):
		if isinstance(input_imagae,torch.Tensor):
			image_tensor = input_imagae.data
		else:
			return input_imagae
		image_numpy = image_tensor.cpu().float().numpy()
		if image_numpy.shape[0] == 1:
			image_numpy = np.tile(image_numpy,(3,1,1))
		image_numpy = (np.transpose(image_numpy,(1,2,0))+1) / 2.0*255.0
	else:
		image_numpy = input_imagae
	return  image_numpy.astype(imtype)

class Logger():
	def __init__(self, n_epochs, batches_epoch):
		self.viz = Visdom()
		self.n_epochs = n_epochs
		self.batches_epoch = batches_epoch
		self.epoch = 1
		self.batch = 1
		self.prev_time = time.time()
		self.mean_period = 0
		self.losses = {}
		self.loss_windows = {}
		self.image_windows = {}


	def log(self, losses=None, images=None):
		self.mean_period += (time.time() - self.prev_time)
		self.prev_time = time.time()

		sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

		for i, loss_name in enumerate(losses.keys()):
			if loss_name not in self.losses:
				self.losses[loss_name] = losses[loss_name]
			else:
				self.losses[loss_name] += losses[loss_name]

			if (i+1) == len(losses.keys()):
				sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
			else:
				sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

		batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
		batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
		sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

		# Draw images
		for image_name, tensor in images.items():
			if image_name not in self.image_windows:
				self.image_windows[image_name] = self.viz.image(tensor2im(tensor).transpose([2,0,1]), opts={'title':image_name})
			else:
				self.viz.image(tensor2im(tensor).transpose([2,0,1]), win=self.image_windows[image_name], opts={'title':image_name})

		# End of epoch
		if (self.batch % self.batches_epoch) == 0:
			# Plot losses
			for loss_name, loss in self.losses.items():
				if loss_name not in self.loss_windows:
					self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
																	opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
				else:
					self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
				# Reset losses for next epoch
				self.losses[loss_name] = 0.0

			self.epoch += 1
			self.batch = 1
			sys.stdout.write('\n')
		else:
			self.batch += 1
