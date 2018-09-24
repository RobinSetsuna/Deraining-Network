from vgg16 import *


def get_perceptual_loss(img_real, img_fake):
    with tf.Graph().as_default():
        session = tf.Session()
        image = tf.placeholder(tf.float32, [None, 224, 224, 3])
        feature = vgg16(image, 'vgg16_weights.npz', session)

        #img1 = imread('laska.png', mode='RGB')
        #img1 = imresize(img1, (224, 224))
        print(img_fake)
        # 概率
        feature_fake = session.run(feature.conv2_2, feed_dict={feature.imgs: img_fake})
        feature_real = session.run(feature.conv2_2, feed_dict={feature.imgs: img_real})

        return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(feature_real, feature_fake))))
