import torch
import os
import time
import numpy as np
import math
from utils import print_results,compute_acc
class Learner(object):

    def __init__(self, args, model, optimizer, scheduler, training_loader, test_loader):

        self.model = model
        self.optim = optimizer

        self.scheduler = None
        if scheduler is not None:
            self.scheduler = scheduler

        self.lr_update_threshold = self.optim.param_groups[0]['lr'] / 100.0

        self.training_loader = training_loader
        self.test_loader = test_loader

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.lambda4 = args.lambda4

        self.cuda = args.cuda



        self.max_grad_norm = args.max_grad_norm

        self.num_epochs = args.num_epochs
        self.start_epoch = 0
        self.global_step = 0
        self.nll_gap = args.nll_gap
        self.valid_gap = args.valid_gap
        self.num_samples = args.num_samples
        self.save_gap = args.save_gap


        self.out_root = args.out_root

        self.checkpoints_dir = os.path.join(self.out_root, "save_models")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.save_dataset_dir = os.path.join(self.out_root, "save_datasets")
        if not os.path.exists(self.save_dataset_dir):
            os.makedirs(self.save_dataset_dir)


    def compute_loss(self, x, ws, ub, mask):

        '''
        :param x:  B x x_dim
        :param ws: B x num_weak_labeler x num_classes
        :param ub: B x num_weak_labeler x num_classes
        :param mask: B x num_weak_labeler x num_classes
        :return:
        '''

        # nll, y: B x num_classes
        y, nll = self.model(x)

        L = ws.size(1)


        # simplex constraint
        constraint_0 = torch.max(-y, torch.zeros_like(y))
        constraint_0 = torch.square(constraint_0)
        constraint_0 = torch.sum(constraint_0, dim=1)


        constraint_1 = torch.max(y - torch.ones_like(y), torch.zeros_like(y))
        constraint_1 = torch.square(constraint_1)
        constraint_1 = torch.sum(constraint_1, dim=1)

        y_sum = torch.sum(y, dim=1)
        constraint_sum = y_sum - torch.ones_like(y_sum)
        constraint_sum = torch.square(constraint_sum)


        # ALL constraint

        y_extend = torch.repeat_interleave(torch.unsqueeze(y, 1), repeats=L, dim=1)
        c = (torch.ones_like(y_extend)-y_extend)*ws + y_extend*(torch.ones_like(y_extend) - ws)

        c = c * mask # mask out NaN weak signals.

        normalizer = torch.sum(mask, dim=0, keepdim=True) + 1e-8
        c = torch.sum(c, dim=0, keepdim=True)
        c = c / normalizer
        ub = torch.mean(ub, dim=0, keepdim=True)
        constraint_ws = torch.max(c - ub, torch.zeros_like(ub))
        constraint_ws = torch.square(constraint_ws)
        constraint_ws = torch.sum(constraint_ws,dim=(1,2))


        return y, torch.mean(nll), torch.mean(constraint_0), torch.mean(constraint_1), \
               torch.mean(constraint_sum), torch.mean(constraint_ws)



    def train(self):

        self.model.train()

        starttime = time.time()
        prev_loss = 0.0

        # run
        num_batchs = len(self.training_loader)
        total_its = self.num_epochs * num_batchs

        for epoch in range(self.start_epoch, self.num_epochs):


            mean_nll = 0.0
            mean_con_0 = 0.0
            mean_con_1 = 0.0
            mean_con_sum = 0.0
            mean_con_ws = 0.0

            for i_batch, batch in enumerate(self.training_loader):

                x = batch["x"]
                y = batch["y"]
                ws = batch["ws"]
                ub = batch["ub"]
                if 'mask' in batch:
                    mask = batch['mask']
                else:
                    mask = torch.ones_like(ws)

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    ws = ws.cuda()
                    ub = ub.cuda()
                    mask = mask.cuda()


                self.model.zero_grad()
                self.optim.zero_grad()

                y, nll, constraint_0, constraint_1, contraint_sum, constraint_ws = self.compute_loss(x, ws, ub, mask)

                loss = nll + \
                       self.lambda1*constraint_0 + \
                       self.lambda2*constraint_1 + \
                       self.lambda3*contraint_sum + \
                       self.lambda4*constraint_ws

                loss.backward()

                if np.isclose(loss.item(), prev_loss) and epoch >= 500:

                    self.valid(epoch, phase="train", save_labels=True)
                    self.valid(epoch, phase="test", save_labels=True)
                    self.save_model(epoch, islatest=False)
                    return

                prev_loss = loss.item()


                # clip grad

                grad_norm = 0
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)


                # vae step
                if math.isnan(grad_norm):
                    continue
                else:
                    self.optim.step()

                mean_nll = mean_nll + nll.data
                mean_con_0 = mean_con_0 + constraint_0.data
                mean_con_1 = mean_con_1 + constraint_1.data
                mean_con_sum = mean_con_sum + contraint_sum.data
                mean_con_ws = mean_con_ws + constraint_ws.data


                # print iteration loss
                currenttime = time.time()
                elapsed = currenttime - starttime
                print(f"Iteration: {self.global_step}/{total_its} \t",
                      f"Epoch: {epoch}/{self.num_epochs} \t",
                      f"Elapsed time: {elapsed:.2f} \t",
                      f"NLL:{nll.data:.5f} \t",
                      f"Cons_0:{constraint_0.data:.5f} \t",
                      f"Cons_1:{constraint_1.data:.5f} \t",
                      f"Cons_sum:{contraint_sum.data:.5f} \t",
                      f"Cons_ws:{constraint_ws.data:.5f} \t",
                      )
                if self.global_step % self.nll_gap == 0:
                    outpath = os.path.join(self.out_root, "loss.txt")
                    variable_list = [self.global_step, elapsed, nll.data, constraint_0.data, constraint_1.data,
                                     contraint_sum.data, constraint_ws.data]
                    print_results(variable_list, outpath)
                self.global_step = self.global_step + 1

            # save every epoch loss
            mean_nll = float(mean_nll / float(num_batchs))
            mean_con_0 = float(mean_con_0 / float(num_batchs))
            mean_con_1 = float(mean_con_1 / float(num_batchs))
            mean_con_sum = float(mean_con_sum / float(num_batchs))
            mean_con_ws = float(mean_con_ws / float(num_batchs))
            outpath = os.path.join(self.out_root, "Epoch_loss.txt")
            currenttime = time.time()
            elapsed = currenttime - starttime
            variable_list = [epoch, elapsed, mean_nll, mean_con_0, mean_con_1, mean_con_sum, mean_con_ws]
            print_results(variable_list, outpath)
            self.save_model(epoch, islatest=True)



            current_lr = self.optim.param_groups[0]['lr']
            if self.scheduler is not None and current_lr > self.lr_update_threshold:
                self.scheduler.step()



            if epoch % self.valid_gap == 0 and epoch > 0:
                if self.training_loader is not None and self.test_loader is not None:

                    save_labels = epoch == (self.num_epochs-1)
                    self.valid(epoch, phase="train", save_labels = save_labels)
                    self.valid(epoch, phase="test", save_labels = save_labels)


            if epoch % self.save_gap == 0 and epoch > 0:
                self.save_model(epoch, islatest=False)


    def save_model(self, epoch, islatest=False):

        if islatest is True:
            outpath = os.path.join(self.checkpoints_dir, "checkpoints_latest.pth.tar")
        else:
            outpath = os.path.join(self.checkpoints_dir, "checkpoints_{}.pth.tar".format(epoch))

        state = {}
        state["epoch"] = epoch
        state["iteration"] = self.global_step
        state["model"] = self.model.state_dict()

        state["optim"] = self.optim.state_dict()

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        else:
            state["scheduler"] = None

        torch.save(state, outpath)


    def valid(self, epoch, phase="train", save_labels=False):

        # phase = train, test

        print("Start valid")

        starttime = time.time()

        mean_nll = 0.0
        mean_con_0 = 0.0
        mean_con_1 = 0.0
        mean_con_sum = 0.0
        mean_con_ws = 0.0
        y_list = []

        if phase == "train":
            dataset = self.training_loader

        else:
            dataset = self.test_loader


        num_batchs = len(dataset)

        pred_list = []
        true_list = []

        with torch.no_grad():
            for i_batch, batch in enumerate(dataset):

                x = batch["x"]
                y = batch["y"]
                ws = batch["ws"]
                ub = batch["ub"]
                if 'mask' in batch:
                    mask = batch['mask']
                else:
                    mask = torch.ones_like(ws)

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    ws = ws.cuda()
                    ub = ub.cuda()
                    mask = mask.cuda()

                pred = self.prediction(x, self.num_samples)
                true_label = y.detach().clone().cpu().numpy()

                for i in range(0, len(pred)):
                    pred_list.append(pred[i])
                    true_list.append(np.argmax(true_label[i]))


                y, nll, constraint_0, constraint_1, contraint_sum, constraint_ws = self.compute_loss(x, ws, ub, mask)


                mean_nll = mean_nll + nll.data
                mean_con_0 = mean_con_0 + constraint_0.data
                mean_con_1 = mean_con_1 + constraint_1.data
                mean_con_sum = mean_con_sum + contraint_sum.data
                mean_con_ws = mean_con_ws + constraint_ws.data



                y = y.detach().clone().cpu().numpy()
                for b in range(0, len(y)):
                    y_list.append(y[b])



                if save_labels == True:
                    y = np.asarray(true_list)
                    y_ = np.asarray(pred_list)
                    x = x.detach().cpu().numpy()
                    data_dict = {"x":x, "y":y, "y_":y_}
                    path = os.path.join(self.save_dataset_dir, "{}_set_{}.npy".format(phase, epoch))
                    np.save(path, data_dict)



            mean_nll = float(mean_nll / float(num_batchs))
            mean_con_0 = float(mean_con_0 / float(num_batchs))
            mean_con_1 = float(mean_con_1 / float(num_batchs))
            mean_con_sum = float(mean_con_sum / float(num_batchs))
            mean_con_ws = float(mean_con_ws / float(num_batchs))
            mean_acc = compute_acc(pred_list, true_list)
            mean_acc = round(mean_acc, 3)


            if phase == 'train':
                outpath = os.path.join(self.out_root, "train_loss.txt")

            elif phase == "valid":
                outpath = os.path.join(self.out_root, "valid_loss.txt")
            else:
                outpath = os.path.join(self.out_root, "test_loss.txt")

            currenttime = time.time()
            elapsed = currenttime - starttime
            variable_list = [epoch, elapsed, mean_acc, mean_nll, mean_con_0, mean_con_1, mean_con_sum, mean_con_ws]
            print_results(variable_list, outpath)


            print("End valid. Elapsed time: {:.2f}".format(time.time()-starttime))



    def prediction(self, x, num_samples):

        with torch.no_grad():
            y_list = []
            for i in range(0, num_samples):
                y, nll = self.model(x)
                y = torch.clip(y, 0, 1)
                y_list.append(y.detach().clone().cpu().numpy())

            pred = np.mean(y_list, axis=0)
            pred = np.argmax(pred, axis=1)


            return pred
