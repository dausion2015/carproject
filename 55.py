import tensorflow as tf
from matplotlib import pyplot as plt
import vgg_preprocessing
filename = 'D:\\BaiduYunDownload\\pj_vechicle\\quiz_train_00000of00004.tfrecord'

filenamequeue = tf.train.string_input_producer([filename])
reader =tf.TFRecordReader()
_,serialized = reader.read(filenamequeue)
features = tf.parse_single_example(serialized,features={
    'image/class/label':tf.FixedLenFeature([],tf.int64),
    'image/encoded':tf.FixedLenFeature([],tf.string),
    'image / height':tf.FixedLenFeature([],tf.int64),
    'image/width':tf.FixedLenFeature([],tf.int64)
    })
image = tf.decode_raw(features['image/encoded'], tf.uint8)
height = features['image / height']
width =  features['image/width']
shape = tf.stack([height,width,tf.constant(3,dtype=tf.int64)])
# image = tf.image.decode_jpeg(features['image/encoded'],3)
image = tf.reshape(image,shape)    # THIS IS IMP
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([height,width,shape,tf.shape(image)]))
sess.close()
# image = tf.decode_raw(features['image/encoded'],tf.uint8)
# image =tf.image.decode_jpeg(features['image/encoded'],3)
# # image = tf.cast(image,tf.uint8)
# # f = plt.figure()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.shape(image)))
# image = tf.cast(image,tf.string)
# plt.imshow(image)
# # image = tf.reshape(image,[224,224,3])
# image = tf.cast(image, tf.float32)
# img = vgg_preprocessing.preprocess_for_train(image,224,224,
#                          resize_side_min=256,
#                          resize_side_max=512)
# plt.imshow(tf.cast(img,tf.uint8))
# label = features['image/class/label'] 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     queue =tf.train.start_queue_runners(sess=sess,coord=coord)
#     fig = plt.figure()
#     for i in range(5):
#         image = sess.run([image,])
#         plt.imshow(image)
        
        
    
