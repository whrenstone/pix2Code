# encoding: utf-8

import tensorflow as tf
from skimage import io
import numpy as np
import argparse
import pdb
import os
import time
import codecs

from generator import gen_batch,myget_images,myBatchGetImages,myBatchGetImagesForInfer
from hyperparams import Hyperparams as hp
from en_decoder import nets
from cnn_nets import cnn

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="inference", choices=["train", "inference"])
parser.add_argument("--layer", default=0, choices=["0", "1", "2", "3"])
parser.add_argument("--atn_type", default="Bahdanau", choices=["Bahdanau", "Luong"])
parser.add_argument("--beamsearch", default=0, choices=["0", "1"])

a = parser.parse_args()

#if a.mode == "train":
	#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
#elif a.mode == "inference":
    #os.environ['CUDA_VISIBLE_DEVICES'] = ""


def train():
    ## read data.
    #predict = tf.equal(tf.cast(pred, tf.int32),tf.cast(_decoder_targets, tf.int32))
    predict = tf.equal(tf.argmax(pred, -1), tf.argmax(decoder_targets, -1))
    #pred_outputs = tf.argmax(pred, -1)
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

    avg_acc = []
    debug_mode = 0;
    for step in range(hp.step_nums):
        if debug_mode==1:
            t = gen_batch(hp.batch_size)
            input_batch, label_batch, target_batch, i_len_batch, l_len_batch = t.next()
        else:
            imgpath ='./TrainImage/'
            target_batch = []; label_batch = []; i_len_batch=[]; l_len_batch= [];
            t = myBatchGetImages(imgpath,hp.batch_size)
            input_batch, label_batch, target_batch, i_len_batch, l_len_batch  = t.next()

        results, _, loss, acc = sess.run([pred,optimizer, cost, accuracy], feed_dict={encoder_inputs: input_batch,
                                                        decoder_labels: label_batch,
                                                        decoder_targets: target_batch,
                                                        encoder_inputs_length: i_len_batch,
                                                        decoder_labels_length: l_len_batch})

        avg_acc.append(acc)

        if step % 10 == 0:
            print ">> Step: %d Loss: %6f Accuracy: %6f atn_layer: %s atn_type: %6s" \
                % (step + 1, loss, acc, a.layer, a.atn_type)

        if step % 1000 == 0:
            vocab_list = hp.string_list

            model_path = save_model_path + "/step_"+ str(step) +"-bz_"+ str(hp.batch_size) \
                        + "-acc_" + str(np.mean(avg_acc[step-100:step])) + ".ckpt"
            saver.save(sess, model_path)

            print(">> save model in %s\n") % model_path

            curStep = int(step)
            txtPath = "inference/inference_"+str(curStep)+'.txt'
            if debug_mode==1:
                txtPath = "inference/inference_Test"+str(curStep)+'.txt'

            for i, result in enumerate(results):
                #print 'target_batch[i]:{}'.format(target_batch[i])
                #import pdb; pdb.set_trace()
                target = "".join( [vocab_list[t] for t in target_batch[i]] )
                #print 'target:{}'.format(target)
                #print 'result:{}'.format(result)
                #import pdb; pdb.set_trace()
                seq = "".join( [vocab_list[r].encode("utf-8") for r in list(result)] )
                #print 'seq:{}'.format(seq)
                with codecs.open(txtPath, "a") as f:
                    f.write('\n target:')
                    f.write(target.encode("utf-8"))
                    f.write('\n Seq:')
                    f.write(seq)



def inference():
    ## read data.
    #with open("vocab.txt", "r") as f:
    #    vocab_list = f.readlines()
    #    vocab_list = [vocab.replace("\n", "") for vocab in vocab_list]
    vocab_list = hp.string_list

    #imgpath ='/media/veilytech/Model/text-detection-ctpn-master/TestImage/others'
    imgpath = './InferenceTest/'
    target_batch = []; label_batch = []; i_len_batch=[];
    t = myBatchGetImagesForInfer(imgpath,hp.batch_size)
    input_batch, label_batch, target_batch, i_len_batch, l_len_batch = t.next()

    results = sess.run(pred, feed_dict={encoder_inputs: input_batch,
                                        encoder_inputs_length: i_len_batch})

    current_time = time.ctime()
    result_list = []
    output_txt_list = []
    output_txt_list.append("\n" + current_time + "\n\n")
    #print'results:',results
    for i, result in enumerate(results):
        #target = "".join(target_batch[i])
        target = "".join([vocab_list[t] for t in target_batch[i]])
        output_txt_list.append(target.encode("utf-8").split(hp.end_token)[0])
        if int(a.beamsearch):
            for index in range(hp.beam_width):
                r_list = []
                for rlt in result:
                    r_list.append(rlt[index])

                seq = "".join([vocab_list[_r] for _r in r_list])

                output_txt = ">> Output: (beamsearch) {}\n".format(seq.encode("utf-8"))
                print output_txt
                output_txt_list.append(seq.encode("utf-8").split(hp.end_token)[0])

        else:
            seq = "".join([vocab_list[r[0]] for r in result])

            output_txt = ">> Output: {}\n".format(seq.encode("utf-8"))
            print output_txt
            output_txt_list.append(seq.encode("utf-8").split(hp.end_token)[0] + "\n")

    with open("inference/inference.txt", "a") as f:
       f.write("\n".join(output_txt_list))

    print ">> inference success.\n"

if __name__ == '__main__':

    encoder_inputs = tf.placeholder(dtype=tf.float32, shape=(None, hp.size[0], hp.size[1]), name="encoder_inputs")
    decoder_labels = tf.placeholder(dtype=tf.int32, shape=(None, None), name="decoder_labels")
    decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name="decoder_targets")

    # encoder_inputs_length: [batch_size]
    # decoder_labels_length: [batch_size]
    encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
    decoder_labels_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_labels_length')

    cnn_output = cnn(encoder_inputs)

    ## train or inference
    if a.mode == "train":
        cost, pred, _decoder_targets = nets(cnn_output, decoder_labels, decoder_targets, \
                        encoder_inputs_length, decoder_labels_length, a.mode, a.layer, a.beamsearch, a.atn_type)

        optimizer = tf.train.AdamOptimizer(hp.lr).minimize(cost)

    elif a.mode == "inference":
        pred = nets(cnn_output, decoder_labels, decoder_targets, \
                    encoder_inputs_length, decoder_labels_length, a.mode, a.layer, a.beamsearch, a.atn_type)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        save_model_path = hp.log_dir + str(a.atn_type) + "/" + str(a.layer)
        print save_model_path
        last_ckpt = tf.train.latest_checkpoint(save_model_path)


        if a.mode == "train":
            if last_ckpt:
                saver.restore(sess, last_ckpt)
                print ">> restore model from %s successful!" % last_ckpt
            train()

        elif a.mode == "inference":
            # restore
            if last_ckpt:
                saver.restore(sess, last_ckpt)
                print ">> restore model from %s successful!" % last_ckpt

            inference()
