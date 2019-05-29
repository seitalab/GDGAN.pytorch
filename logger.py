import os
import pickle
from collections import defaultdict

import torchvision.utils as vutils
from tensorboardX import SummaryWriter

writer = SummaryWriter()


class Logger(object):
    def __init__(self, config, summary_dir, result_dir):
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        summary_path = os.path.join(summary_dir,
                                    config.date_str + '.txt')
        with open(summary_path, 'w') as f:
            for k, v in vars(config).items():
                text = k + ': ' + str(v)
                print(text, file=f)

        self.result_dir = result_dir
        self.histories = defaultdict(list)

    def add_scalar(self, tag, value, n_iter):
        writer.add_scalar(tag, value, n_iter)
        self.histories[tag].append(value)

    def add_scalars(self, tag, dic, n_iter):
        writer.add_scalars(tag, dic, n_iter)
        for k, v in dic.items():
            self.histories[tag + '/' + k].append(v)

    def add_image(self, tag, img, n_iter):
        writer.add_image(tag, img, n_iter)
        img_dir = os.path.join(self.result_dir, tag)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(img_dir, '%03d.png' % (n_iter))
        vutils.save_image(img, img_path)

    def add_histogram(self, tag, data, n_iter):
        writer.add_histogram(tag, data, n_iter)

    def add_pr_curve(self, tag, ys, pred_ys, n_iter):
        writer.add_pr_curve(tag, ys, pred_ys, n_iter)

    def save_history(self):
        history_path = os.path.join(self.result_dir, 'history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.histories, f)
