from __future__ import division
import argparse
from model import IDCGAN
from tkinter import *
from tkinter.messagebox import askyesno
import tkinter.filedialog
from PIL import Image, ImageTk
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='deraining', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=224, help='then crop to this size')
parser.add_argument('--k', dest='k', type=int, default=64)
parser.add_argument('--k_2', dest='k_2', type=int, default=48)
parser.add_argument('--input_c_dim', dest='input_c_dim', type=int, default=3)
parser.add_argument('--output_c_dim', dest='output_c_dim', type=int, default=3)
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=False, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--sample_size', dest='sample_size', type=int, default=1, help='sample size')

args = parser.parse_args()

def hello():
    print("Hello")


class MainWindow(object):
    def exits(self):
        ans = askyesno(title='注意', message='确定要关闭窗口?')
        if ans:
            self.root.destroy()
        else:
            return

    def xz(self):
        if self.flag == 1:
            self.label_img.destroy()
            self.label_img_real.destroy()
            self.label_img_fake.destroy()
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            self.pic = filename
        else:
            askyesno(title='注意', message='未选择图片')

    def start(self):
        self.notice = tkinter.Label(self.root, text="正在处理...")
        self.notice.pack()
        self.root.after(100, self.runtest)

    def runtest(self):
        self.flag = 1
        self.notice.destroy()

        self.label_img = tkinter.Label(self.root, text="已完成。上图为去雨后图片，下图为原始图片")
        self.label_img.pack()

        with tf.Session() as sess:
            base_dir = self.model.test(self.pic, sess)
            dir_test = base_dir + '//test.png'
            dir_test_real = base_dir + "//test_real.png"

            img = Image.open(dir_test)
            photo_fake = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
            self.label_img_fake = tkinter.Label(self.root, image=photo_fake)
            self.label_img_fake.place(x=0, y=0)
            self.label_img_fake.pack(side='top')

            img_real = Image.open(dir_test_real)
            photo_real = ImageTk.PhotoImage(img_real)  # 用PIL模块的PhotoImage打开
            self.label_img_real = tkinter.Label(self.root, image=photo_real)
            self.label_img_real.place(x=0, y=0)
            self.label_img_real.pack(side='top')


        self.root.mainloop()

    def __init__(self):
        self.flag = 0
        self.root = Tk()
        self.root.title("图像去雨")
        self.root.geometry("700x500+400+200")
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_separator()
        menubar.add_cascade(label="选择图片", command=self.xz)
        menubar.add_cascade(label="启动", command=self.start)
        menubar.add_cascade(label="退出", command=self.exits)
        self.root.config(menu=menubar)
        with tf.Session() as sess:
            self.model = IDCGAN(sess, image_size=args.fine_size, batch_size=args.batch_size,
                           output_size=args.fine_size, dataset_name=args.dataset_name, sample_size=args.sample_size,
                           checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, L1_lambda=args.L1_lambda,
                           k=args.k, k_2=args.k_2, input_c_dim=args.input_c_dim, output_c_dim=args.output_c_dim)
            self.model()
        self.root.mainloop()


if __name__ == '__main__':
    MainWindow = MainWindow()
