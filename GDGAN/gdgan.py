# PyTorch
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils

# libraries
import os
import numpy as np

# networks
from GDGAN.networks.stage1.model import generator as g1
from GDGAN.networks.stage1.model import discriminator as d1
from GDGAN.networks.stage2.model import generator as g2
from GDGAN.networks.stage2.model import discriminator as d2

# metrics
from metrics.fid.fid_score import fid
from metrics.inception_score import inception_score

# dataset
import datasets

# logs
from logger import Logger

# utils
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class GDGAN(object):
    def __init__(self, config):
        # common config
        self.seed = config.seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # logs
        self.test_every = config.test_every
        self.save_every = config.save_every

        # experiment details
        self.date_str = config.date_str

        # custom config
        # general
        self.dataset_name = config.dataset
        self.root_path = config.root_path
        self.label_path = config.label_path
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.nc = config.nc
        self.nz = config.nz
        self.n_critic = config.n_critic
        self.lambda_ = config.lambda_
        self.lrG = config.lrG
        self.lrD = config.lrD
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # loss
        self.mse_loss_flag = config.mse_loss
        self.lambda_G1_1 = config.lambda_G1_1
        self.lambda_D1_1 = config.lambda_D1_1
        self.lambda_D1_2 = config.lambda_D1_2
        self.lambda_G2_1 = config.lambda_G2_1
        self.lambda_G2_2 = config.lambda_G2_2
        self.lambda_G2_3 = config.lambda_G2_3
        self.lambda_D2_1 = config.lambda_D2_1
        self.lambda_D2_2 = config.lambda_D2_2
        self.lambda_D2_3 = config.lambda_D2_3

        # G1 and D1
        self.epoch1 = config.epoch1
        self.y1_indices = config.y1_indices
        self.G1_path = config.G1_path
        self.D1_path = config.D1_path

        # G2 and D2
        self.epoch2 = config.epoch2
        self.y2_indices = config.y2_indices

        # ys and z for visualization
        self.sample_num = 16
        self.z_ = torch.rand((self.sample_num, self.nz)).to(device)
        self.y1_ = self.generate_class_vec(len(self.y1_indices),
                                           self.sample_num)
        self.y2_ = self.generate_class_vec(len(self.y2_indices),
                                           self.sample_num)

        # dataset and dataloader
        if self.dataset_name == 'chest-xray':
            transform = transforms.Compose([transforms.Grayscale(),
                                            transforms.Resize(self.image_size),
                                            transforms.ToTensor()])
            self.dataset = \
                datasets.ChestXrayDataset(root=self.root_path,
                                          image_list_file=self.label_path,
                                          train=True,
                                          transform=transform)
            dataset_indices = list(range(len(self.dataset)))
            # not using val dataset for classifier
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    dataset_indices[:-self.dataset.val_num])
            self.dataloader = \
                torch.utils.data.DataLoader(self.dataset,
                                            sampler=train_sampler,
                                            batch_size=self.batch_size,
                                            num_workers=4,
                                            drop_last=True)

        # networks
        self.G1 = g1(self.nc, self.nz, image_size=self.image_size,
                     class_num=len(self.y1_indices))
        self.D1 = d1(self.nc, image_size=self.image_size,
                     class_num=len(self.y1_indices))
        self.G1.apply(utils.init_weights)
        self.D1.apply(utils.init_weights)
        if torch.cuda.device_count() > 1:
            print("use multiple GPUs in parallel")
            self.G1 = nn.DataParallel(self.G1)
            self.D1 = nn.DataParallel(self.D1)
        self.G1 = self.G1.to(device)
        self.D1 = self.D1.to(device)

        self.G2 = g2(self.nc, image_size=self.image_size,
                     class_num=len(self.y2_indices))
        self.D2 = d2(self.nc, image_size=self.image_size,
                     class_num=len(self.y2_indices))
        self.G2.apply(utils.init_weights)
        self.D2.apply(utils.init_weights)
        if torch.cuda.device_count() > 1:
            print("use multiple GPUs in parallel")
            self.G2 = nn.DataParallel(self.G2)
            self.D2 = nn.DataParallel(self.D2)
        self.G2 = self.G2.to(device)
        self.D2 = self.D2.to(device)

        # best state dict for best model
        self.G1_best_state_dict = None
        self.D1_best_state_dict = None
        self.G2_best_state_dict = None
        self.D2_best_state_dict = None
        self.best_inception_score1 = 0
        self.best_inception_score2 = 0
        self.best_fid_score1 = 100000
        self.best_fid_score2 = 100000

        # optimizers
        self.G1_optimizer = optim.Adam(self.G1.parameters(),
                                       lr=self.lrG,
                                       betas=(self.beta1, self.beta2))
        self.D1_optimizer = optim.Adam(self.D1.parameters(),
                                       lr=self.lrD,
                                       betas=(self.beta1, self.beta2))
        self.G2_optimizer = optim.Adam(self.G2.parameters(),
                                       lr=self.lrG,
                                       betas=(self.beta1, self.beta2))
        self.D2_optimizer = optim.Adam(self.D2.parameters(),
                                       lr=self.lrD,
                                       betas=(self.beta1, self.beta2))

        # losses
        self.BCEWL_loss = nn.BCEWithLogitsLoss()
        if self.mse_loss_flag:
            self.mse_loss = nn.MSELoss()

        # metrics
        self.fid = fid(self.dataloader, device)

        # logs
        summary_dir = os.path.join('summary', config.dataset)
        result_dir = os.path.join('result',
                                  config.dataset,
                                  config.date_str)
        self.logger = Logger(config,
                             summary_dir,
                             result_dir)

    def run(self):
        if self.G1_path != '' and self.D1_path != '':
            self.load1()
        else:
            self.stage1_iter = 0
            print('start stage1 train')
            for epoch in range(self.epoch1):
                print('epoch:', epoch + 1)
                self.train1(epoch)
                self.visualize1(epoch)
                if (epoch + 1) % self.test_every == 0:
                    self.test1(epoch)
                if (epoch + 1) % self.save_every == 0:
                    self.save1(epoch)

        self.stage2_iter = 0
        print('start stage2 train')
        for epoch in range(self.epoch2):
            print('epoch:', epoch + 1)
            self.train2(epoch)
            self.visualize2(epoch)
            if (epoch + 1) % self.test_every == 0:
                self.test2(epoch)
            if (epoch + 1) % self.save_every == 0:
                self.save2(epoch)

        print('finish train')
        self.finalize()

    def train1(self, epoch):
        self.G1.train()
        self.D1.train()
        for i, (x_, y_) in enumerate(self.dataloader):
            self.stage1_iter += 1

            y_ = y_[:, self.y1_indices]
            x_, y_ = x_.to(device), y_.to(device)
            z_ = torch.rand((self.batch_size, self.nz)).to(device)

            # update discriminator
            self.D1_optimizer.zero_grad()

            D_real, C_real = self.D1(x_)
            D_real_loss = -torch.mean(D_real)
            D_C_real_loss = self.lambda_D1_2 * self.BCEWL_loss(C_real, y_)

            G_ = self.G1(z_, y_)
            D_fake, C_fake = self.D1(G_)
            D_fake_loss = torch.mean(D_fake)
            D_C_fake_loss = self.lambda_D1_1 * self.BCEWL_loss(C_fake, y_)

            # gradient penalty
            alpha = torch.rand((self.batch_size, 1, 1, 1)).to(device)
            x_hat = alpha * x_.detach() + (1 - alpha) * G_.detach()
            x_hat.requires_grad_()
            pred_hat, _ = self.D1(x_hat)
            gradients = grad(
                outputs=pred_hat,
                inputs=x_hat,
                grad_outputs=torch.ones(pred_hat.size()).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            gradient_penalty = self.lambda_ * \
                ((gradients.view(gradients.size()[0], -1).norm(2, 1)
                    - 1) ** 2).mean()

            D_loss = D_real_loss + D_C_real_loss + \
                D_fake_loss + D_C_fake_loss + gradient_penalty

            self.logger.add_scalar('data/stage1/D_loss',
                                   D_loss.item(),
                                   self.stage1_iter)
            scalars_dic = {'D_real_loss': D_real_loss.item(),
                           'D_C_real_loss': D_C_real_loss.item(),
                           'D_fake_loss': D_fake_loss.item(),
                           'D_C_fake_loss': D_C_fake_loss.item(),
                           'gradient_penalty': gradient_penalty.item()}
            self.logger.add_scalars('data/stage1/D_loss_detail',
                                    scalars_dic,
                                    self.stage1_iter)

            D_loss.backward()
            self.D1_optimizer.step()

            if (i + 1) % self.n_critic == 0:
                # update generator
                self.G1_optimizer.zero_grad()

                G_ = self.G1(z_, y_)
                D_fake, C_fake = self.D1(G_)
                G_fake_loss = -torch.mean(D_fake)
                G_C_fake_loss = self.lambda_G1_1 * self.BCEWL_loss(C_fake, y_)
                G_loss = G_fake_loss + G_C_fake_loss

                self.logger.add_scalar('data/stage1/G_loss',
                                       G_loss.item(),
                                       self.stage1_iter)
                scalars_dic = {'G_fake_loss': G_fake_loss.item(),
                               'G_C_fake_loss': G_C_fake_loss.item()}
                self.logger.add_scalars('data/stage1/G_loss_detail',
                                        scalars_dic,
                                        self.stage1_iter)

                G_loss.backward()
                self.G1_optimizer.step()

    def load1(self):
        self.G1.load_state_dict(torch.load(self.G1_path))
        self.D1.load_state_dict(torch.load(self.D1_path))

    def visualize1(self, epoch):
        self.G1.eval()
        with torch.no_grad():
            samples = self.G1(self.z_, self.y1_).detach()
            imgs = vutils.make_grid(samples)
            self.logger.add_image('images/stage1', imgs, epoch+1)

        for name, param in self.G1.named_parameters():
            self.logger.add_histogram('params/G1/'+name,
                                      param.clone().cpu().data.numpy(),
                                      epoch+1)
        for name, param in self.D1.named_parameters():
            self.logger.add_histogram('params/D1/'+name,
                                      param.clone().cpu().data.numpy(),
                                      epoch+1)

    def test1(self, epoch):
        self.G1.eval()
        self.D1.eval()

        test_num = 10000
        steps = 200
        batch_size = test_num // steps
        imgs = np.empty((test_num, 1, self.image_size, self.image_size))
        start, end = 0, 0
        with torch.no_grad():
            for i in range(steps):
                start = end
                end = start + batch_size

                z_ = torch.rand((batch_size, self.nz)).to(device)
                y1_ = self.generate_class_vec(len(self.y1_indices), batch_size)
                G_ = self.G1(z_, y1_)
                imgs[start:end] = G_.detach().cpu().numpy()

        # inception score
        inception_value, _ = inception_score(imgs, device)
        self.logger.add_scalar('data/stage1/inception_score',
                               inception_value,
                               epoch+1)
        # fid score
        fid_value = self.fid.calculate_score(imgs)
        self.logger.add_scalar('data/stage1/fid_score', fid_value, epoch+1)

        if (self.best_inception_score1 < inception_value
                or self.best_fid_score1 > fid_value):
            print('best model updated')
            self.best_inception_score1 = inception_value
            self.best_fid_value1 = fid_value
            self.G1_best_state_dict = self.G1.state_dict()
            self.D1_best_state_dict = self.D1.state_dict()

    def save1(self, epoch):
        save_dir = os.path.join('result',
                                self.dataset_name,
                                self.date_str,
                                'models',
                                'stage1')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G1.state_dict(),
                   os.path.join(save_dir, 'G_epoch%03d.pkl' % (epoch + 1)))
        torch.save(self.D1.state_dict(),
                   os.path.join(save_dir, 'D_epoch%03d.pkl' % (epoch + 1)))

    def train2(self, epoch):
        if (self.G1_best_state_dict is not None
                and self.D1_best_state_dict is not None):
            self.G1.load_state_dict(self.G1_best_state_dict)
            self.D1.load_state_dict(self.D1_best_state_dict)
        self.G1.eval()
        self.D1.eval()
        self.G2.train()
        self.D2.train()

        for i, (x_, y_) in enumerate(self.dataloader):
            self.stage2_iter += 1
            y_ = y_[:, self.y2_indices]
            x_, y_ = x_.to(device), y_.to(device)
            z_ = torch.rand((self.batch_size, self.nz)).to(device)
            y1_ = self.generate_class_vec(len(self.y1_indices),
                                          self.batch_size)

            # update discriminator
            self.D2_optimizer.zero_grad()
            D_real, C_real = self.D2(x_)
            D_real_loss = -torch.mean(D_real)
            D_C_real_loss = self.lambda_D2_2 * self.BCEWL_loss(C_real, y_)

            G1_ = self.G1(z_, y1_)
            G_ = self.G2(G1_, y_)
            D_fake, C_fake = self.D2(G_)
            D_fake_loss = torch.mean(D_fake)
            D_C_fake_loss = self.lambda_D2_1 * self.BCEWL_loss(C_fake, y_)

            # gradient penalty
            alpha = torch.rand((self.batch_size, 1, 1, 1)).to(device)

            x_hat = alpha * x_.detach() + (1 - alpha) * G_.detach()
            x_hat.requires_grad_()
            pred_hat, _ = self.D2(x_hat)
            gradients = grad(
                outputs=pred_hat,
                inputs=x_hat,
                grad_outputs=torch.ones(pred_hat.size()).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            gradient_penalty = self.lambda_ * \
                ((gradients.view(gradients.size()[0], -1).norm(2, 1)
                    - 1) ** 2).mean()

            _, c = self.D1(G_)
            D_D1_loss = self.lambda_D2_3 * self.BCEWL_loss(c, y1_)

            D_loss = D_real_loss + D_C_real_loss + \
                D_fake_loss + D_C_fake_loss + gradient_penalty + D_D1_loss

            self.logger.add_scalar('data/stage2/D_loss',
                                   D_loss.item(),
                                   self.stage2_iter)
            scalars_dic = {'D_real_loss': D_real_loss.item(),
                           'D_C_real_loss': D_C_real_loss.item(),
                           'D_fake_loss': D_fake_loss.item(),
                           'D_C_fake_loss': D_C_fake_loss.item(),
                           'D_D1_loss': D_D1_loss.item(),
                           'gradient_penalty': gradient_penalty.item()}
            self.logger.add_scalars('data/stage2/D_loss_detail',
                                    scalars_dic,
                                    self.stage2_iter)
            D_loss.backward()
            self.D2_optimizer.step()

            if (i + 1) % self.n_critic == 0:
                # update generator
                self.G2_optimizer.zero_grad()

                G1_ = self.G1(z_, y1_)
                G_ = self.G2(G1_, y_)
                D_fake, C_fake = self.D2(G_)
                G_fake_loss = -torch.mean(D_fake)
                G_C_fake_loss = self.lambda_G2_1 * self.BCEWL_loss(C_fake, y_)

                _, c = self.D1(G_)
                G_D1_loss = self.lambda_G2_2 * self.BCEWL_loss(c, y1_)

                G_loss = G_fake_loss + G_C_fake_loss + G_D1_loss
                if self.mse_loss_flag:
                    mse_loss = self.lambda_G2_3 * \
                        self.mse_loss(G_, G1_.detach())
                    G_loss += mse_loss

                self.logger.add_scalar('data/stage2/G_loss',
                                       G_loss.item(),
                                       self.stage2_iter)
                scalars_dic = {'G_fake_loss': G_fake_loss.item(),
                               'G_C_real_loss': G_C_fake_loss.item(),
                               'G_D1_loss': G_D1_loss.item()}
                if self.mse_loss_flag:
                    scalars_dic['mse_loss'] = mse_loss.item()
                self.logger.add_scalars('data/stage2/G_loss_detail',
                                        scalars_dic,
                                        self.stage2_iter)
                G_loss.backward()
                self.G2_optimizer.step()

    def visualize2(self, epoch):
        self.G1.eval()
        self.G2.eval()
        with torch.no_grad():
            G1_ = self.G1(self.z_, self.y1_)
            samples = self.G2(G1_, self.y2_).detach()
            imgs = vutils.make_grid(samples)
            self.logger.add_image('images/stage2', imgs, epoch+1)

        for name, param in self.G2.named_parameters():
            self.logger.add_histogram('params/G2/'+name,
                                      param.clone().cpu().data.numpy(),
                                      epoch+1)
        for name, param in self.D2.named_parameters():
            self.logger.add_histogram('params/D2/'+name,
                                      param.clone().cpu().data.numpy(),
                                      epoch+1)

    def test2(self, epoch):
        self.G1.eval()
        self.D1.eval()
        self.G2.eval()
        self.D2.eval()

        test_num = 10000
        steps = 200
        batch_size = test_num // steps
        imgs = np.empty((test_num, 1, self.image_size, self.image_size))
        start, end = 0, 0
        with torch.no_grad():
            for i in range(steps):
                start = end
                end = start + batch_size

                z_ = torch.rand((batch_size, self.nz)).to(device)
                y1_ = self.generate_class_vec(len(self.y1_indices), batch_size)
                y2_ = self.generate_class_vec(len(self.y2_indices), batch_size)
                G1_ = self.G1(z_, y1_)
                G_ = self.G2(G1_, y2_)
                imgs[start:end] = G_.detach().cpu().numpy()

        # inception score
        inception_value, _ = inception_score(imgs, device)
        self.logger.add_scalar('data/stage2/inception_score',
                               inception_value,
                               epoch+1)
        # fid score
        fid_value = self.fid.calculate_score(imgs)
        self.logger.add_scalar('data/stage2/fid_score', fid_value, epoch+1)

        if (self.best_inception_score2 < inception_value
                or self.best_fid_score2 > fid_value):
            print('best model updated')
            self.best_inception_score2 = inception_value
            self.best_fid_value2 = fid_value
            self.G2_best_state_dict = self.G2.state_dict()
            self.D2_best_state_dict = self.D2.state_dict()

    def save2(self, epoch):
        save_dir = os.path.join('result',
                                self.dataset_name,
                                self.date_str,
                                'models',
                                'stage2')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G2.state_dict(),
                   os.path.join(save_dir, 'G_epoch%03d.pkl' % (epoch + 1)))
        torch.save(self.D2.state_dict(),
                   os.path.join(save_dir, 'D_epoch%03d.pkl' % (epoch + 1)))

    def finalize(self):
        self.logger.save_history()
        model_dir = os.path.join('result',
                                 self.dataset_name,
                                 self.date_str,
                                 'models')
        # save best models
        if self.G1_best_state_dict is not None:
            torch.save(self.G1_best_state_dict,
                       os.path.join(model_dir, 'stage1', 'G_best.pkl'))
        if self.D1_best_state_dict is not None:
            torch.save(self.D1_best_state_dict,
                       os.path.join(model_dir, 'stage1', 'D_best.pkl'))
        if self.G2_best_state_dict is not None:
            torch.save(self.G2_best_state_dict,
                       os.path.join(model_dir, 'stage2', 'G_best.pkl'))
        if self.D2_best_state_dict is not None:
            torch.save(self.D2_best_state_dict,
                       os.path.join(model_dir, 'stage2', 'D_best.pkl'))

    def generate_class_vec(self, class_num, batch_size):
        y_base_ = torch.eye(class_num + 1)[:, :class_num]
        y_ = y_base_.repeat((batch_size // (class_num + 1)) + 1, 1)
        return y_[:batch_size].to(device)
