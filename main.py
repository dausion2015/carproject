from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from utils import extract

from utils import data_loader
import shutil
# import caffe
# import vgg_preprocessing
from datetime import datetime
import cv2
import py_generate_bbox
from datetime import datetime

slim = tf.contrib.slim
# def im2double(im):
# 	return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)



def norm_image(img):
    
    shape = img.get_shape().as_list()
    if len(shape) == 3:
        for i in range(shape[-1]):
            temp = img[:, :, i]
            img[:,:,i] = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
            img[:,:,i] = np.around(img[:,:,i]*255).astype(np.uint8)
    else:
        temp = img[:, :]
        img[:,:] = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
        img[:,:] = np.around(img[:,:,i]*255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)

def image_sparate(img_cam,img_orig):

    shape = img_cam.get_shape().as_list()
    for i in range(shape[-1]):
        tmp = 0.2*img_orig+0.7*img_cam[:,:,i]
        scipy.misc.imsave(os.path.join(args.out_dir, 'test_heatmap_top5_{}.jpg'.format(str(i))), img_cam[:,:,i])
        scipy.misc.imsave(os.path.join(args.out_dir, 'bindtest_heatmap_top5_{}.jpg'.format(str(i))), tmp)


def test(args):

    if args.imgpath is None: raise SystemExit('imgpath is None')
    image = scipy.misc.imread(args.imgpath, mode='RGB')
    image = scipy.misc.imresize(image, (224, 224))
    image = np.expand_dims(image, 0)
    # assert image.shape == (1, 224, 224, 3)

    # image_ph = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))

    # model = VGG16_GAP(args, num_batches=None, train_mode=False)
    # model.build(image_ph)

    # saver = tf.train.Saver()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())

    # if args.modelpath is not None:
    #     print ('Using model: {}'.format(args.modelpath))
    #     saver.restore(sess, args.modelpath)
    # else:
    #     print ('Using pretrained caffemodel')

    # feed_dict = {image_ph: image}
    # logits, CAM_conv_resize, CAM_fc = sess.run([
    #     model.cam_fc, model.cam_conv_resize, model.cam_fc_value], feed_dict=feed_dict)
    # pred = np.argmax(logits)
    # acc = tf.nn.softmax(logits)
    # # pdb.set_trace()
    # acesend = np.argsort(logits)
    # acesend_list = list(acesend)
    # acesend_list.reverse()
    # desecend = acesend
    # top_5 = desecend[:5]
    # cam_map_list = []
    # for i in range(top_5):   #CAM_conv_resize [224,224,1024]
    #     CAM_h_map = np.matmul(CAM_conv_resize.reshape(-1, 1024), CAM_fc.transpose()[desecend[i]].transpose())
    #     CAM_h_map = np.reshape(CAM_h_map,[224,224])
    #     cam_map_list.append(CAM_h_map)
    # CAM_heatmap2 = tf.concat(cam_map_list,axis=2)
    # CAM_heatmap = np.matmul(CAM_conv_resize.reshape(-1, 1024), CAM_fc.transpose()[pred])
    # CAM_heatmap = np.reshape(CAM_heatmap, [224, 224])
    # CAM_heatmap = norm_image(CAM_heatmap)
    # CAM_heatmap2 = norm_image(CAM_heatmap2)
    
    # image = np.squeeze(image)
    # curHeatMap = image*0.2+CAM_heatmap*0.7
    # scipy.misc.imsave(os.path.join(args.out_dir, 'test_heatmap.jpg'), CAM_heatmap)
    # scipy.misc.imsave(os.path.join(args.out_dir,'bindtest_heatmap.jpg'),curHeatMap)
    # image_sparate(CAM_heatmap2,image)

    # with open(os.path.join(args.datast_dir2,'labels.txt'), 'r', encoding='utf8') as f:
    #     h = ((i[0], i[1]) for i in [i.split(':') for i in f.read().strip().split('\n')])
    #     dic = dict(h)
    # class_name = dic[pred]
    # sess.close()
    print ('Bye')
    # return class_name,acc

def train(flag,num,args):
    
    queue_loader = data_loader(flag,num,batch_size=args.bsize, num_epochs=args.ep,dataset_dir=args.dataset_dir)
  
    with slim.arg_scope(inception_utils.inception_arg_scope()):
        logits, end_points = inception_v3.inception_v3(queue_loader.images,
                        num_classes=764,
                        is_training=flag,
                        dropout_keep_prob=0.8,
                        min_depth=16,
                        depth_multiplier=1.0,
                        prediction_fn=slim.softmax,
                        spatial_squeeze=True,
                        reuse=None,
                        create_aux_logits=True,
                        scope='InceptionV3',
                        global_pool=True)

        total_logist =  logits+end_points['AuxLogits']
        loss_op = tf.losses.sparse_softmax_cross_entropy(queue_loader.labels, total_logist)
        reg_loss_op = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(loss_op, reg_loss_op)
        train_op = tf.train.RMSPropOptimizer(args.lr).minimize(total_loss)   
        accuracy = tf.equal(tf.argmax(total_logist, 1), queue_loader.labels)
        tf.summary.scalar('cross entropy loss', loss_op)
        tf.summary.scalar('regularization loss', reg_loss_op)
        tf.summary.scalar('total loss', total_loss)
        merged_op = tf.summary.merge_all()

        var_to_restore = slim.get_variables_to_restore(exclude=['InceptionV3/AuxLogits/Conv2d_1b_1x1',
                                                            'InceptionV3/Logits/Conv2d_1b_1x1/Conv2d_1c_1x1'])
        var_to_init = slim.get_variables_to_restore(include=['InceptionV3/AuxLogits/Conv2d_1b_1x1',
                                               'InceptionV3/Logits/Conv2d_1b_1x1/Conv2d_1c_1x1'])
      
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        writer = tf.summary.FileWriter(args.trainlog, sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                                                  
                                                             
        saver = tf.train.Saver(var_list=var_to_restore,max_to_keep=3) 
        

        if len(os.listdir(args.modelpath)) == 0:  
            saver.restore(var_to_restore,os.path.join(args.dataset_dir2,'model.ckpt'))
        else:
            var_to_restore = slim.get_model_variables() 
            saver.restore(var_to_restore,tf.train.latest_checkpoint(args.modelpath))
       
   
    print ('Start training')
    print ('batch size: %d, epoch: %d, initial learning rate: %.3f' % (args.bsize, args.ep, args.lr))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

   
   
    try:
        ep = 0
        g_step = tf.train.get_or_create_global_step()
        correct_all = 0

        while not coord.should_stop():  
            # model.logits = sess.run([model.cam_fc])
            # xloss = sess.run([model.xen_loss_op])
            # rloss = sess.run([model.reg_loss_op])
            # loss = sess.run([model.loss_op,model.correct_op])
            # correct = sess.run([model.loss_op,model.correct_op])
            # _ = sess.run([train_op])
            # summary = sess.run([merged_op])
            # g_step = sess.run([g_step])
            
            logit, xloss, rloss, loss, correct, _, summary,g_s = sess.run([
                logits,loss_op, reg_loss_op, total_loss,accuracy, train_op, merged_op, g_step])   #
           
            
            writer.add_summary(summary, g_s)
            
            # print('argmax logits',np.argmax(logits,1),sess.run(queue_loader.labels))
            # print('correct.sum :  ',correct.sum())
            correct_all += correct.sum()
            
           
            
            if g_s % 40 == 0:
                print ('epoch: %2d, globle_step: %3d, xloss: %.2f, rloss: %.2f, loss: %.3f' % (ep, g_s, xloss, rloss, loss))
                
            if g_s/queue_loader.num_batches > 1:
                print ('epoch: %2d, step: %3d, xloss: %.2f, rloss: %.2f, loss: %.3f, epoch %2d done.' %
                        (ep+1, g_s, xloss, rloss, loss, ep+1))
                print ('EPOCH %2d ACCURACY: %.2f%%.' % (ep, correct_all * 100/queue_loader.num_batches))
                saver.save(sess,os.path.join(args.modelpath,'model.ckpt'), global_step=g_s)
                ep += 1        
                correct_all = 0
                continue

    except tf.errors.OutOfRangeError:
        print ('\nDone training, epoch limit: %d reached.' % (ep))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print ('Done')
def evalidate(flag,num,args):
    with tf.Graph().as_default():
        queue_loader = data_loader(False,num,batch_size=args.bsize,num_epochs=args.ep,dataset_dir=args.dataset_dir)
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits, end_points = inception_v3.inception_v3(queue_loader.images,
                            num_classes=764,
                            is_training=flag,
                            dropout_keep_prob=0.8,
                            min_depth=16,
                            depth_multiplier=1.0,
                            prediction_fn=slim.softmax,
                            spatial_squeeze=True,
                            reuse=None,
                            create_aux_logits=True,
                            scope='Inception3',
                            global_pool=True)
        
        
        
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "accuracy": slim.metrics.accuracy(np.argmax(logits), queue_loader.labels)
        })

        for name,value in names_to_values.items():
            op_name = 'eval_{}'.format(name)
            op = tf.summary.scalar(op_name,value)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            slim.evaluation.evaluate_once(
                                            master='',
                                            checkpoint_path=args.modelpath,
                                            logdir=args.evallog,
                                            um_evals= np.ceil(queue_loader.num_batches/args.bsize),
                                            eval_op=list(names_to_updates.values()),
                                            variables_to_restore=slim.get_variables_to_restore())
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # saver = tf.train.Saver()
    # saver.restore(tf.train.latest_checkpoint(args.modelpath))


if __name__ == '__main__':
    print('########################## start running ###############################')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weightpath', metavar='', type=str, default=None, help='trained tensorflow weight path.')
    parser.add_argument('--dataset_dir',default='/data/ai100/quiz-w7' ,type=str,help='dataset here')
    parser.add_argument('--dataset_dir2',default='/data/dausion2015/vehicle-project' ,type=str,help='dataset here')
    # parser.add_argument('--train', action='store_true', help='set this to train.')
    # parser.add_argument('--test', action='store_true', help='set this to test.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=5, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=32, help='batch size.')
    parser.add_argument('--modelpath', metavar='', type=str, default='output/checkpoint', help='trained tensorflow model path.')
    parser.add_argument('--imgpath', type=str, default=os.path.dirname(__file__)+'test.jpg', help='Test image path.')
    parser.add_argument('--output_dir', type=str, default='/output', help=None)
    parser.add_argument('--pretrain', type=str, default='/output/pretrain', help=None)
    parser.add_argument('--evallog', type=str, default='/output/evallog', help=None)
    parser.add_argument('--trainlog', type=str, default='/output/trainlog', help=None)
    args, unparsed = parser.parse_known_args()
   
    w_path = os.path.dirname(os.path.abspath(__file__))
    print(w_path,'current work space!!!!!!')
    #os.makedirs(os.path.join(w_path,'output/pretrain'))
    os.makedirs(args.pretrain)
    if os.path.exists(args.pretrain):
        print('##########################################################/output/pretrain has been faound')
    #os.makedirs(os.path.join(w_path,'output/checkpoint'))
    os.makedirs(args.modelpath)
    if os.path.exists(args.modelpath):
        print('##########################################################output/checkpoint has been faound')
    #os.makedirs(os.path.join(w_path,'output/checkpoint'))
   # os.makedirs(os.path.join(w_path,'output/evallog'))
    os.makedirs(args.evallog)
    if os.path.exists(args.evallog):
        print('##########################################################o/output/evallog has been faound')
    #os.makedirs(os.path.join(w_path,'output/trainlog'))
    os.makedirs(args.trainlog)
    if os.path.exists(args.evallog):
        print('##########################################################/output/trainlog has been faound')
    shutil.copy(os.path.join(w_path,'test.jpg'),os.path.join(args.output_dir,'test.jpg'))
    # shutil.copy(os.path.join(w_path,'deploy_vgg16CAM.prototxt'),os.path.join(args.pretrain,'deploy_vgg16CAM.prototxt'))
    # shutil.copy(os.path.join(w_path,'vgg16CAM_train_iter_90000.caffemodel'),os.path.join(args.pretrain,'vgg16CAM_train_iter_90000.caffemodel'))
    # pre_model = os.path.join(args.dataset_dir2,'deploy_vgg16CAM.prototxt')
    # pre_weight = os.path.join(args.dataset_dir2,'vgg16CAM_train_iter_90000.caffemodel')
    # pre_model = 'dataset'
    # pre_weight = 'dataset'
    # npy_path = extract(pre_model,pre_weight)
    # print(npy_path)


    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    for i in range(4):  
        # if args.train:
            # for i in range(num):
        print("########################## {}tims training #############################".format(str(i+1)))
        tf.logging.info("########################## begain trainning #################################")
        train(True,i,args)
        train(True,i,args)
        
        tf.logging.info("########################## ending trainning  #################################")
        print("########################## {}tims training end #############################".format(str(i+1)))
    for i in range(4):
        print("########################## {}tims evalidting #############################".format(str(i+1)))
        tf.logging.info("########################## begain evalidting #################################")    
        evalidate(False,i,args)
        tf.logging.info("########################## ending evalidting #################################")
        print("########################## {}tims evalidting end #############################".format(str(i+1)))

    # tf.logging.info("########################## begain testing #################################")
    # print("########################## begain testing #################################")
    # start = datetime.now()
    # class_name,acc = test(args)
    # print("########################## testing end #################################")
    # tf.logging.info("########################## testing end #################################")
    # print('[[[[[[[[[[[[[ cost time{}seconds'.format(start-datetime.now()))
    # curHeatMapFile = os.path.join(args.out_dir,'test_heatmap.jpg')
    # curImgFile = os.path.join(args.output_dir,'test.jpg')
    # curBBoxFile = os.path.join(args.dataset_dir2,'heatmap_6.txt') 
    # rectg = py_generate_bbox.generate_bbox(curHeatMapFile=curHeatMapFile,curImgFile =curImgFile,curBBoxFile=curBBoxFile)
    # img = cv2.imread(curImgFile)
    # # for i in range(rectg.shape[0]):
    # cv2.rectangle(img,tuple(rectg[0,0:2]),tuple(rectg[0,2:]),(255,0,0), 3)
    # # 标注文本
    # font = cv2.FONT_HERSHEY_SUPLEX
    # cv2.putText(img, class_name+'  '+str(acc*100)+'%', (rectg[0,0]+20,rectg[0,3]), font, 3, (0,0,255), 1)
    # cv2.imwrite(os.path.join(args.output_dir,'bbox_img.jpg', img))
