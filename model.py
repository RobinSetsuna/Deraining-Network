from __future__ import division
import os
import time
from glob import glob
from ops import *
from utils import *
from vgg16 import *


class IDCGAN(object):
    def __init__(self, sess, image_size, batch_size, sample_size, output_size, L1_lambda, k, k_2,
                 input_c_dim, output_c_dim, dataset_name,
                 checkpoint_dir, sample_dir):
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.sample_dir = sample_dir
        self.Flag = 0

        self.k = k
        self.k_2 = k_2

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization
        self.d_bn_h1 = BatchNorm(name='d_bn_h1')
        self.d_bn_h2 = BatchNorm(name='d_bn_h2')
        self.d_bn_h3 = BatchNorm(name='d_bn_h3')
        self.g_bn_h2 = BatchNorm(name='g_bn_h2')
        self.g_bn_h3 = BatchNorm(name='g_bn_h3')
        self.g_bn_h4 = BatchNorm(name='g_bn_h4')
        self.g_bn_h5 = BatchNorm(name='g_bn_h5')
        self.g_bn_h6 = BatchNorm(name='g_bn_h6')
        self.g_bn_h7 = BatchNorm(name='g_bn_h7')
        self.g_bn_h8 = BatchNorm(name='g_bn_h8')
        self.g_bn_h9 = BatchNorm(name='g_bn_h9')
        self.g_bn_h10 = BatchNorm(name='g_bn_h10')
        self.g_bn_h11 = BatchNorm(name='g_bn_h11')
        self.g_bn_h12 = BatchNorm(name='g_bn_h12')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

    def __call__(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        # 无雨图片REAL_B
        self.real_B = self.real_data[:, :, :, :self.input_c_dim]
        # 有雨图片REAL_A
        self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        # 生成(假)的去雨图片FAKE_B----对应REAL_A

        self.fake_B = self.generator(self.real_A)
        self.vgg1 = vgg16(self.real_B, 'vgg16_weights.npz', self.sess)
        self.feature_real = self.vgg1.conv2_2
        self.vgg2 = vgg16(self.fake_B, 'vgg16_weights.npz', self.sess)
        self.feature_fake = self.vgg2.conv2_2

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss_pixel = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(self.fake_B - self.real_B))))
        self.perceptual_loss = tf.sqrt(tf.reduce_sum(tf.square(self.feature_fake - self.feature_real))) / 224 * 224 * 3
        self.g_loss = self.g_loss_pixel + self.perceptual_loss
        self.d_loss = 0.5 * self.d_loss_real + 0.5 * self.d_loss_fake
        self.loss = self.g_loss + 6.6e-3 * self.d_loss
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.vars = self.d_vars + self.g_vars

        self.saver = tf.train.Saver()

    def load_weights(self, weight_file, sess):
        print("enter_load_weights")
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    # 随机读取取样图象
    def load_random_samples(self, args):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file, fine_size=args.fine_size) for sample_file in data]

        if self.is_grayscale:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    # 取样模型
    def sample_model(self, sample_dir, epoch, idx, args):
        sample_images = self.load_random_samples(args=args)
        sample_images_p = sample_images[:, :, :, :self.input_c_dim]
        save_images(sample_images_p, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_real.png'.format(sample_dir, epoch, idx))
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_fake.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=(args.beta1/4)).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)
        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.loss, var_list=self.vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch = [load_data(batch_file, fine_size=args.fine_size, is_test=False) for batch_file in batch_files]
                if self.is_grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                '''
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={self.real_data: batch_images})

                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)
                '''
                self.sess.run(optim, feed_dict={self.real_data: batch_images})

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 50) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, args)

                if np.mod(counter, 50) == 1:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, reuse):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            print("discri")
            h0 = prelu(conv2d(image, self.k_2, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h0_conv'), reuse=reuse,
                       name='d_h0_prelu')
            h1 = prelu(self.d_bn_h1(conv2d(h0, 2 * self.k_2, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h1_conv')),
                       reuse=reuse,
                       name='d_h1_prelu')
            h2 = prelu(self.d_bn_h2(conv2d(h1, 4 * self.k_2, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h2_conv')),
                       reuse=reuse,
                       name='d_h2_prelu')
            h3 = prelu(self.d_bn_h3(conv2d(h2, 8 * self.k_2, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h3_conv')),
                       reuse=reuse,
                       name='d_h3_prelu')
            h4 = conv2d(h3, 1, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h4_conv')
            # h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h5_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            # image is (256 x 256 x input_c_dim)
            # Conv
            h1 = conv2d(image, self.k, name='g_h1_conv')

            h2 = self.g_bn_h2(conv2d(prelu(h1, name='g_h1_prelu'), self.k, name='g_h2_conv'))

            h3 = self.g_bn_h3(conv2d(prelu(h2, name='g_h2_prelu'), self.k, name='g_h3_conv'))

            h4 = self.g_bn_h4(conv2d(prelu(h3, name='g_h3_prelu'), self.k, name='g_h4_conv'))

            h5 = self.g_bn_h5(conv2d(prelu(h4, name='g_h4_prelu'), int(self.k / 2), name='g_h5_conv'))

            h6 = self.g_bn_h6(conv2d(prelu(h5, name='g_h5_prelu'), 1, name='g_h6_conv'))

            # Deconv
            h7 = deconv2d(tf.nn.relu(h6, name='g_h6_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, int(self.k / 2)], name='g_h7_deconv')
            h7 = self.g_bn_h7(h7)

            h8 = deconv2d(tf.nn.relu(h7, name='g_h7_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, self.k], name='g_h8_deconv')
            h8 = self.g_bn_h8(h8)
            h8 = tf.concat([h8, h4], 3)

            h9 = deconv2d(tf.nn.relu(h8, name='g_h8_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, self.k], name='g_h9_deconv')
            h9 = self.g_bn_h9(h9)

            h10 = deconv2d(tf.nn.relu(h9, name='g_h9_relu'), [self.batch_size, self.image_size,
                                                              self.image_size, self.k], name='g_h10_deconv')
            h10 = self.g_bn_h10(h10)
            h10 = tf.concat([h10+h2], 3)

            h11 = deconv2d(tf.nn.relu(h10, name='g_h10_relu'), [self.batch_size, self.image_size,
                                                                self.image_size, self.k], name='g_h11_deconv')
            h11 = self.g_bn_h11(h11)

            h12 = deconv2d(tf.nn.relu(h11, name='g_h11_relu'), [self.batch_size, self.image_size,
                                                                self.image_size, 3], name='g_h12_deconv')
            h12 = self.g_bn_h12(h12)
            h12 = tf.concat([h12 + image], 3)

            return tf.nn.tanh(h12)

    def sampler(self, image):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            # image is (256 x 256 x input_c_dim)
            # Conv
            h1 = conv2d(image, self.k, name='g_h1_conv')

            h2 = self.g_bn_h2(conv2d(prelu(h1, name='g_h1_prelu'), self.k, name='g_h2_conv'))

            h3 = self.g_bn_h3(conv2d(prelu(h2, name='g_h2_prelu'), self.k, name='g_h3_conv'))

            h4 = self.g_bn_h4(conv2d(prelu(h3, name='g_h3_prelu'), self.k, name='g_h4_conv'))

            h5 = self.g_bn_h5(conv2d(prelu(h4, name='g_h4_prelu'), int(self.k / 2), name='g_h5_conv'))

            h6 = self.g_bn_h6(conv2d(prelu(h5, name='g_h5_prelu'), 1, name='g_h6_conv'))

            # Deconv
            h7 = deconv2d(tf.nn.relu(h6, name='g_h6_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, int(self.k / 2)], name='g_h7_deconv')
            h7 = self.g_bn_h7(h7)

            h8 = deconv2d(tf.nn.relu(h7, name='g_h7_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, self.k], name='g_h8_deconv')
            h8 = self.g_bn_h8(h8)
            h8 = tf.concat([h8, h4], 3)

            h9 = deconv2d(tf.nn.relu(h8, name='g_h8_relu'), [self.batch_size, self.image_size,
                                                             self.image_size, self.k], name='g_h9_deconv')
            h9 = self.g_bn_h9(h9)

            h10 = deconv2d(tf.nn.relu(h9, name='g_h9_relu'), [self.batch_size, self.image_size,
                                                              self.image_size, self.k], name='g_h10_deconv')
            h10 = self.g_bn_h10(h10)
            h10 = tf.concat([h10 + h2], 3)

            h11 = deconv2d(tf.nn.relu(h10, name='g_h10_relu'), [self.batch_size, self.image_size,
                                                                self.image_size, self.k], name='g_h11_deconv')
            h11 = self.g_bn_h11(h11)

            h12 = deconv2d(tf.nn.relu(h11, name='g_h11_relu'), [self.batch_size, self.image_size,
                                                                self.image_size, 3], name='g_h12_deconv')
            h12 = self.g_bn_h12(h12)
            h12 = tf.concat([h12 + image], 3)

            return tf.nn.tanh(h12)

    def save(self, checkpoint_dir, step):
        model_name = "idcgan.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir, sess):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            print(os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, sample_files, sess):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # load testing input
        print("Loading testing images ...")
        sample_file = load_data(sample_files, fine_size=224, is_test=True)
        sample_file = sample_file.reshape([1, 224, 224, 6])

        sample_file_p = sample_file[:, :, :, :self.input_c_dim]
        save_images(sample_file_p, [self.batch_size, 1],
                    './{}/test_real.png'.format('test1'))

        sample = np.array(sample_file).astype(np.float32)

        if self.load(self.checkpoint_dir, sess):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        samples = sess.run(
            self.fake_B_sample,
            feed_dict={self.real_data: sample}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/test.png'.format('test1'))

        dir = './test1'
        return dir

