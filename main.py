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
# import inception_utils
# import inception_v4

import resnet_v2

from utils import data_loader
import shutil
# import caffe
# import vgg_preprocessing
from datetime import datetime
import cv2
from tensorflow.contrib import slim
import math

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

def train(flag,args):
    tf.reset_default_graph()
    
    queue_loader = data_loader(flag,batch_size=args.bsize, num_epochs=args.ep,dataset_dir=args.dataset_dir)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_101(queue_loader.images, 764, is_training=flag)
                     
        g_step = tf.Variable(0,trainable=False,dtype=tf.int64,name='g_step')
        # g_step = tf.train.create_global_step()
        decay_steps = int(math.floor(43971 /args.bsize*2.0))
        lr = tf.train.exponential_decay(args.lr,global_step=g_step,decay_steps=decay_steps,decay_rate=0.94,staircase=True)
       
        # loss_AuxLogits = tf.losses.sparse_softmax_cross_entropy( labels=queue_loader.labels, logits=end_points['AuxLogits'], weights=0.4)
        total_loss = tf.losses.sparse_softmax_cross_entropy( labels=queue_loader.labels, logits=logits,  weights=1.0)
        
        # total_loss = tf.losses.get_total_loss(add_regularization_losses=False)

#         train_op = tf.train.RMSPropOptimizer(lr).minimize(total_loss) 
  
        labels = queue_loader.labels
        correct = tf.equal(tf.argmax(logits, 1), labels)
      
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
     
        # tf.summary.scalar('AuxLogits loss', loss_AuxLogits)
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.scalar('total loss', total_loss)
        tf.summary.scalar('learnning_rate', lr)
        merged_op = tf.summary.merge_all()
        # for var in slim.get_model_variables():
        #     if var.op.name.startswith("Variabl"):
        #         print(var)
        
        
        with tf.variable_scope('Adam_vars'):
            # optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op,global_step=global_step)
            optimizer = tf.train.AdamOptimizer(lr)
            grads_vars_tuple_list = optimizer.compute_gradients(total_loss)
            grads,vars = zip(*grads_vars_tuple_list)
            new_grads,_ = tf.clip_by_global_norm(grads,5)
            new_grads_vars_tuple_list = zip(new_grads,vars)
            train_op = optimizer.apply_gradients(new_grads_vars_tuple_list, global_step=g_step)
            # optimizer = tf.train.AdamOptimizer(lr)
            # grads_and_vars = optimizer.compute_gradients(loss_op)
            # for i,(grad,var) in enumerate(grads_and_vars):
            #     if grad is not None:
            #         grads_and_vars[i] = (tf.clip_by_norm(grad,5),var)
            # train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)


        if len(os.listdir(args.modelpath)) > 0:
            var_to_restore = slim.get_model_variables()
            ckpt_path = tf.train.latest_checkpoint(args.modelpath)
            print('###########################ckpt_path',ckpt_path)
      
        else:
            # var_to_restore = slim.get_variables_to_restore(exclude=['InceptionV3/AuxLogits/Conv2d_1b_1x1',
            #                                                 'InceptionV3/Logits/Conv2d_1b_1x1/Conv2d_1c_1x1'])
            var_to_restore = slim.get_variables_to_restore(exclude=[
                                                            'resnet_v2_101/logits','Adam_vars','g_step'])
                                                            # 'InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/beta/Adam',
                                                            # 'InceptionV4/AuxLogits/Conv2d_1b_1x1/weights/Adam'    
            # Aux_logit = slim.get_variables_to_restore(include=['InceptionV4/AuxLogits/Aux_logits'])
            # Logit = slim.get_variables_to_restore(include=['InceptionV4/Logits/Logits'])   
            # adam = slim.get_variables_to_restore(include=['Adam_vars'])                                                             
            ckpt_path = tf.train.latest_checkpoint(args.dataset_dir2)
            print('###########################ckpt_path',ckpt_path)
        # Aux_logit_init = tf.variables_initializer(Aux_logit)
        # Logit_init = tf.variables_initializer(Logit)
        # adam_init = tf.variables_initializer(adam)
        init_func = slim.assign_from_checkpoint_fn(ckpt_path,var_to_restore) 
      
        saver = tf.train.Saver(max_to_keep=3)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        writer = tf.summary.FileWriter(args.trainlog, sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # sess.run(Aux_logit_init)
        # sess.run(Logit_init)
        # sess.run(adam_init)
        init_func(sess)
        print('###################################### restore checkpoint sucessful#######################')

       
   
        print ('Start training')
        print ('batch size: %d, epoch: %d, initial learning rate: %.3f' % (args.bsize, args.ep, args.lr))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

   
    
        try:
            ep = 1
            correct_all = 0
            start = datetime.now()
            while not coord.should_stop():     
                # logit, loss, correct_list, _, summary,acc,g_s= sess.run([
                #     logits,loss_op,correct, train_op, merged_op,accuracy,global_step])
                logit, loss, correct_list, _, summary, acc, g_s, l = sess.run([
                    logits, total_loss,correct, train_op, merged_op, accuracy, g_step, lr])   
                 
                writer.add_summary(summary, g_s)
                
                # print('argmax logits',np.argmax(logits,1),sess.run(queue_loader.labels))
                # print('correct.sum :  ',correct.sum())
                correct_all += correct_list.sum()
                # step_accuracy = correct_list.sum()*100.0/args.bsize
            
                
                if g_s % 10 == 0:
                    end_time = datetime.now()
                    print ('epoch: %2d, globle_step: %3d,accuracy : %.2f%%,loss: %.3f, cost time : %s sec'
                            % (ep, g_s,acc*100.0, loss,end_time-start))
                    start = datetime.now()
                if g_s != 0 and g_s % queue_loader.num_batches == 0:
                    print ('EPOCH %2d is end, ACCURACY: %.2f%%.' % (ep, correct_all * 100.0/(queue_loader.num_batches*args.bsize)))
                    
                    
                    saver.save(sess,os.path.join(args.modelpath,'model.ckpt'), global_step=g_s)
                    ep += 1        
                    correct_all = 0  
                    
                   
        except tf.errors.OutOfRangeError:
            print ('\nDone training, epoch limit: %d reached.' % (ep))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()
        print ('Done')
def evalidate(flag,args):
    with tf.Graph().as_default():
        queue_loader = data_loader(False,batch_size=args.bsize,num_epochs=args.ep,dataset_dir=args.dataset_dir)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(queue_loader.images,num_classes=764,is_training=flag) 
                           
        
        
        
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "accuracy": slim.metrics.accuracy(np.argmax(logits,1), queue_loader.labels)
        })

        for name,value in names_to_values.items():
            op_name = 'eval_{}'.format(name)
            op = tf.summary.scalar(op_name,value)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            slim.evaluation.evaluate_once(
                                            master='',
                                            checkpoint_path=args.modelpath,
                                            logdir=args.evallog,
                                            num_evals= queue_loader.num_batches,
                                            eval_op=list(names_to_updates.values()),
                                            variables_to_restore=slim.get_variables_to_restore())
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # saver = tf.train.Saver()
    # saver.restore(tf.train.latest_checkpoint(args.modelpath))


if __name__ == '__main__':
    print('###########################prev-outputexit',os.listdir('prev-output'))
    print('########################## start running ###############################')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weightpath', metavar='', type=str, default=None, help='trained tensorflow weight path.')
    parser.add_argument('--dataset_dir',default='/data/ai100/quiz-w7' ,type=str,help='dataset here')
    parser.add_argument('--dataset_dir2',default='/data/dausion2015/vehicle-project' ,type=str,help='dataset here')
    # parser.add_argument('--train', action='store_true', help='set this to train.')
    # parser.add_argument('--test', action='store_true', help='set this to test.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=20, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=16, help='batch size.')
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
    if os.path.exists(args.pretrain):
        print('##########################################################/output/pretrain has been faound')
    else:
        os.makedirs(args.pretrain)
    #os.makedirs(os.path.join(w_path,'output/checkpoint'))
    
    if os.path.exists(args.modelpath):
        print('##########################################################output/checkpoint has been faound')
    else:
        os.makedirs(args.modelpath)
    #os.makedirs(os.path.join(w_path,'output/checkpoint'))
   # os.makedirs(os.path.join(w_path,'output/evallog'))
    
    if os.path.exists(args.evallog):
        print('##########################################################o/output/evallog has been faound')
    else:
        os.makedirs(args.evallog)
    #os.makedirs(os.path.join(w_path,'output/trainlog'))
    
    if os.path.exists(args.evallog):
        print('##########################################################/output/trainlog has been faound')
    else:
        os.makedirs(args.trainlog)
    
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
    
    print("########################## training #############################")
    tf.logging.info("########################## begain trainning #################################")
    train(True,args)
    tf.logging.info("########################## ending trainning  #################################")
    
    
    
    # tf.logging.info("########################## begain evalidting #################################")    
    # evalidate(False,args)
    # tf.logging.info("########################## ending evalidting #################################")
   

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
    
