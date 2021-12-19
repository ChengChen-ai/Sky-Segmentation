import os
import torch
import torch.nn as nn
import torch.optim as optim
import ops
from collections import OrderedDict
from torch.autograd import Variable
from Models.enhance import Enhance_Net,Decom_Net
from Models.generator import Generator
from Models.vgg16 import Vgg16
from Models.Discriminator import Discriminator
from util.image_pool import ImagePool

import torch.nn.functional as F

class SSMNet(nn.Module):
    def __init__(
        self, params, use_gpu
    ):
        super(SSMNet,self).__init__()

        self.params = params
        self.save_path = params.save_path
        self.num_classes = params.num_classes

        # containers for data and labels
        self.shape_A= (params.batch_size, params.input_nc, params.height, params.width)
        self.shape_B = (params.batch_size, params.output_nc, params.height, params.width)
        self.input_A = torch.Tensor(*self.shape_A)
        self.input_B = torch.Tensor(*self.shape_B)
        self.real_A = Variable(torch.Tensor(*self.shape_A))
        self.real_B = Variable(torch.Tensor(*self.shape_B))
        self.feature_map = Variable(torch.Tensor(*self.shape_A))
        self.fake_sky = Variable(torch.Tensor(*self.shape_A))
        self.result_map = Variable(torch.Tensor(*self.shape_A))
        self.image_label = Variable(torch.Tensor(*self.shape_B))

        ##models
        self.netE = Enhance_Net("netE")
        self.netD = Decom_Net("netD", params.nums_layer)
        self.netG = Generator('netG', params.input_nc, params.num_filter, params.output_nc, params.nums_layer)
        self.netVgg16 = Vgg16("netVgg", False)
        self.netD_sky = Discriminator(name='divsky', input_dim=params.input_nc, num_filter=params.num_filter,
                                      output_dim=1)

        # criterions
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionL1L = torch.nn.L1Loss()

        D_sky_size = self.netD_sky.forward(self.feature_map).size()
        self.fake_label_sky = Variable(torch.zeros(D_sky_size))
        self.real_label_sky = Variable(torch.Tensor(D_sky_size).fill_(0.9))

        # optimizers
        if params.train:
            self.fakeAPoll = ImagePool(50)
            self.netE_optim = optim.Adam(self.netE.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netD_optim = optim.Adam(self.netD.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netG_optim = optim.Adam(self.netG.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netD_sky_optim = optim.Adam(self.netD_sky.parameters(), lr=params.lr,
                                                 betas=(params.beta1, 0.999))

        if params.cuda:
            if use_gpu:
                print("Suceccesful GPU!!!!!!!!")
                self.netD = self.netD.cuda(0)
                self.netE = self.netE.cuda(0)
                self.netG = self.netG.cuda(0)
                self.netVgg16 = self.netVgg16.cuda(0)
                self.netFog_F = self.netFog_F.cuda(0)
                self.netD_sky = self.netD_sky.cuda(0)

                self.input_A = self.input_A.cuda(0)
                self.input_B = self.input_B.cuda(0)
                self.real_A = self.real_A.cuda(0)
                self.real_B = self.real_B.cuda(0)
                self.feature_map = self.feature_map.cuda(0)
                self.result_map = self.result_map.cuda(0)
                self.fake_sky = self.fake_sky.cuda(0)
                self.fake_label_sky = self.fake_label_sky.cuda(0)
                self.real_label_sky = self.real_label_sky.cuda(0)

                self.criterionMSE = self.criterionMSE.cuda(0)
                self.criterionL1L = self.criterionL1L.cuda(0)
            else:
                print("No GPU!!!!!!!!!!")
                raise NotImplementedError

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test_model(self):
        """test model
        """
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A = self.netFog_F.Gamma_corrected(self.real_A).to(torch.float)
        R_a, I_a = self.netD(self.real_A)
        self.R_a = R_a
        self.I_a = torch.cat([I_a, I_a, I_a], 1)
        self.image_label = torch.cat([I_a, I_a, I_a], 1)
        I_a = torch.max(input=self.I_a, dim=1, keepdim=True)[0]
        I_a_hat = self.netE(R_a, I_a)
        self.I_a_hat = torch.cat([I_a_hat, I_a_hat, I_a_hat], 1)
        self.feature_map = R_a
        out3 = self.netG(self.feature_map)

        self.result_map = out3 + self.I_a_hat
        self.result_map = torch.where(self.result_map>=torch.max(self.result_map)*0.15,torch.ones_like(self.result_map),torch.zeros_like(self.result_map))

    def backward_basic_D(self, netD, real, fake, real_label, fake_label):

        # real log(D_A(B))
        output = netD.forward(real)
        loss_D_real = self.criterionMSE(output, real_label)

        # fake log(1 - D_A((G_A(A)))
        output2 = netD.forward(fake.detach())
        loss_D_fake = self.criterionMSE(output2, fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        return loss_D

    ##D_Loss_fog
    def backward_D_sky(self):
        fake_sky = self.fakeAPoll.query(self.fake_sky)
        self.netD_sky_optim.zero_grad()
        self.loss_D_sky = self.backward_basic_D(self.netD_sky, self.real_B, fake_sky, self.real_label_sky, self.fake_label_sky)
        self.netD_sky_optim.step()

    def backward_D(self):
        self.netD_optim.zero_grad()
        self.real_A = self.netFog_F.Gamma_corrected(self.real_A).to(torch.float)
        real_B = torch.where(self.real_B >= 0.9, torch.ones_like(self.real_B), self.real_A)
        R_a, I_a = self.netD(self.real_A)
        R_b, I_b = self.netD(real_B)

        self.R_a = R_a.detach()
        self.R_b = R_b.detach()
        self.I_a = torch.cat([I_a, I_a, I_a], 1).detach()
        self.I_b = torch.cat([I_b, I_b, I_b], 1).detach()

        loss_reconst_dec = self.criterionL1L(self.real_A, R_a.mul(I_a)) \
                           + self.criterionL1L(real_B, R_b.mul(I_b)) \
                           + 0.001 * self.criterionL1L(self.real_A, R_b.mul(I_a)) \
                           + 0.001 * self.criterionL1L(real_B, R_a.mul(I_b))
        loss_ivref = 0.01 * self.criterionL1L(R_a, R_b)
        loss_dec = loss_reconst_dec + loss_ivref
        smooth_loss_a = ops.get_gradients_loss(I_a, R_a)
        smooth_loss_b = ops.get_gradients_loss(I_b, R_b)
        loss_dec_last = loss_dec + 0.1 * smooth_loss_a + 0.1 * smooth_loss_b


        self.loss_dec = loss_dec_last
        self.loss_dec.backward(retain_graph=True)
        self.netD_optim.step()

    def backward_E(self):
        self.netE_optim.zero_grad()

        real_B = torch.where(self.real_B >= 0.9, torch.ones_like(self.real_B), self.real_A)
        R_aa, I_aa = self.R_a,torch.max(input=self.I_a, dim=1, keepdim=True)[0]
        self.feature_map = self.netE(R_aa, I_aa)

        loss_reconst_enh = self.criterionL1L(real_B, self.feature_map)

        self.loss_enh = loss_reconst_enh
        self.loss_enh.backward(retain_graph=True)
        self.netE_optim.step()

    def backward_G(self):
        self.fake_sky = self.netG(self.feature_map.detach())
        output = self.netD_sky(self.fake_sky)
        self.loss_D_gan = self.criterionMSE(output,self.real_label_sky)
        self.loss_gan = self.criterionL1L(self.real_B, self.fake_sky)

        with torch.no_grad():
            f_real_1, f_real_2 = self.netVgg16(self.real_B)
            f_fake_1, f_fake_2 = self.netVgg16(self.fake_sky)

            loss1 = self.criterionMSE(f_real_1,f_fake_1)
            loss2 = self.criterionMSE(f_real_2,f_fake_2)

        self.loss_vgg = loss1 + loss2
        self.loss_totall = self.loss_gan + self.loss_vgg + self.loss_D_gan

        self.netG_optim.zero_grad()
        self.loss_totall.backward()
        self.netG_optim.step()

    def optimize_parameters_separate(self):
        self.forward()
        self.backward_D()
        self.backward_E()
        self.backward_G()
        self.backward_D_sky()

    def optimize_parameters(self):
        self.optimize_parameters_separate()

    def get_AB_images_triple(self):
        return torch.cat((self.real_A.data, self.R_a.data, self.real_B.data, self.R_b.data,
                         self.feature_map.data, self.fake_sky.data))

    def get_AB_images_triple_test(self):
        return torch.cat((self.real_A.data, self.R_a.data,
                          self.I_a_hat.data, self.feature_map.data,
                          self.result_map.data))

    def set_inputs(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def set_inputs_test(self, input):
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)

    def get_errors(self):
        loss_dec = self.loss_dec.item ()
        loss_enh = self.loss_enh.item ()
        loss_G = self.loss_gan.item ()
        loss_D = self.loss_D_sky.item()
        loss_V = self.loss_vgg.item ()
        loss = self.loss_totall.item ()


        errors = OrderedDict([('loss', loss), ('loss_dec', loss_dec), ('loss_enh', loss_enh),
                            ('loss_G',loss_G),('loss_D',loss_D),('loss_V',loss_V)])

        return errors


    def get_errors_string(self):

        errors = self.get_errors()
        errors_str = ''
        for k, v in errors.items():
            errors_str += '{}={:.4f} '.format(k, v)
        return errors_str

    def save_parameters(self, epoch):
        model_file_netD = os.path.join(self.save_path, 'model_netD_{}.pth'.format(epoch))
        model_file_netE = os.path.join(self.save_path, 'model_netE_{}.pth'.format(epoch))
        model_file_netG = os.path.join(self.save_path, 'model_netG_{}.pth'.format(epoch))

        torch.save(self.netD.state_dict(), model_file_netD)
        torch.save(self.netE.state_dict(), model_file_netE)
        torch.save(self.netG.state_dict(), model_file_netG)



    def load_parameters(self, epoch):
        print('loading model parameters from epoch {}...'.format(epoch))
        map2device = lambda storage, loc: storage

        model_file_netD = os.path.join(self.save_path, 'model_netD_{}.pth'.format(epoch))
        model_file_netE = os.path.join(self.save_path, 'model_netE_{}.pth'.format(epoch))
        model_file_netG = os.path.join(self.save_path, 'model_netG_{}.pth'.format(epoch))

        self.netD.load_state_dict(torch.load(model_file_netD, map_location=map2device))
        self.netE.load_state_dict(torch.load(model_file_netE, map_location=map2device))
        self.netG.load_state_dict(torch.load(model_file_netG, map_location=map2device))

        print('model parameters loaded successfully')
