import tensorflow as tf

import os
import numpy as np
import tensorflow as tf
import time
import itertools
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt

# PARSE ARGUMENTS
# in the order: SEED_DIM, SEED_DIM_BIN, SEED_DIM_D2
[SEED_DIM, SEED_DIM_BIN, SEED_DIM_D2] = sys.argv[1:4]
exp_name = None
if len(sys.argv[1:]) > 3:
    exp_name = sys.argv[4]
SEED_DIM = int(SEED_DIM)
SEED_DIM_BIN = int(SEED_DIM_BIN)
SEED_DIM_D2 = int(SEED_DIM_D2)

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# params
BATCH = 128
lambd = 10
dsteps = 6
train_epochs = 9 #66

load_from_checkpoint = False
if exp_name is not None:
    load_from_checkpoint = True

if exp_name is None:
    from datetime import datetime
    t = datetime.now()
    exp_name = "hpsearchcnn_short/WGANGP_discrete2" + "seed" + str(SEED_DIM) + "_binseed" + str(SEED_DIM_BIN) \
               + "_disseed" + str(SEED_DIM_D2) + "_" + t.strftime('%Y_%m_%d_%H_%M_%S')
    import os
    dir = os.getcwd() + "/" + exp_name
    if not os.path.exists(dir):
        os.makedirs(dir)

load_model_name=None

def write_to_comment_file(comment_file, text):
    with open(comment_file, "a") as f:
        f.write(text)

comments = "WGAN_GP" + \
           "\nCNN archi" + \
           "semibinary z + 4-step z"
comment_file = exp_name + "/comments.txt"
write_to_comment_file(comment_file, comments)

params = "SEED_DIM, " + str(SEED_DIM) + "\n" + \
         "SEED_DIM_BIN, " + str(SEED_DIM_BIN) + "\n" + \
         "SEED_DIM_D2, " + str(SEED_DIM_D2) + "\n" + \
         "dsteps, " + str(dsteps) + "\n" + \
         "train_epochs, " + str(train_epochs)
params_file = exp_name + "/params.csv"
write_to_comment_file(params_file, params)


################## GRAPH DEFINITION ###################""

tf.reset_default_graph()
if 'session' in globals():
    session.close()
session = tf.Session()

# some layers
def dense(x, num_units, reuse, add_bias=True, nonlinearity=None, scope=""):
    with tf.variable_scope(scope, reuse=reuse):
        V = tf.get_variable('V', [int(x.get_shape()[1]), num_units], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
        x = tf.matmul(x, V)
        if add_bias is True:
            b = tf.get_variable('b', [1, num_units], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        x = x + b
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def conv2d(x, num_filters, reuse, filter_size=[3, 3],
           stride=[1, 1], pad='SAME', add_bias=True, nonlinearity=None,
           scope=""):
    with tf.variable_scope(scope, reuse=reuse):
        V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
        x = tf.nn.conv2d(x, V, [1]+stride+[1], pad)
        if add_bias is True:
            b = tf.get_variable('b', [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = x + b
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def conv2d_tr(x, num_filters, reuse, filter_size=[3, 3],
              stride=[1, 1], pad='SAME', add_bias=True,
              nonlinearity=None, scope=""):
    with tf.variable_scope(scope, reuse=reuse):
        xs = [int(x.get_shape()[i]) for i in range(4)]
        if pad=='SAME':
            target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
        else:
            target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
        V = tf.get_variable('V', filter_size+[num_filters, int(x.get_shape()[-1])], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
        x = tf.nn.conv2d_transpose(x, V, target_shape, [1]+stride+[1], pad)
        if add_bias is True:
            b = tf.get_variable('b', [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)
            x = x + b
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

# architecture
def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak*x)

def generator(in_z_bin, in_z_d2, in_z, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        in_z = tf.concat([in_z_bin, in_z_d2, in_z], 1)
        h0 = dense(in_z, num_units=128*7*7,
                   reuse=reuse,
                   nonlinearity=lrelu,
                   scope="h0")
        h0 = tf.reshape(h0, [-1, 7, 7, 128])
        h1 = conv2d_tr(h0, num_filters=64,
                       reuse=reuse, filter_size=[5, 5], stride=[2, 2],
                       nonlinearity=lrelu, scope="h1")
        h2 = conv2d_tr(h1, num_filters=1,
                       reuse=reuse, filter_size=[5, 5], stride=[2, 2],
                       nonlinearity=tf.sigmoid, scope="h2")
        h2 = tf.reshape(h2, [-1, 28, 28, 1])
        return h2

def discriminator(x_in,  reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        x_in = tf.reshape(x_in, [-1, 28, 28, 1])
        h0 = conv2d(x_in, num_filters=64,
                    reuse=reuse, filter_size=[5, 5], stride=[2, 2],
                    nonlinearity=lrelu, scope="h0")
        h1 = conv2d(h0, num_filters=128,
                    reuse=reuse, filter_size=[5, 5], stride=[2, 2],
                    nonlinearity=lrelu, scope="h1")
        h1 = tf.reshape(h1, [-1, 7*7*128])
        h2 = dense(h1, num_units=1,
                   reuse=reuse,
                   nonlinearity=None, scope="h2")
        h2 = tf.reshape(h2, [-1])
        return h2

# placeholders
z_rand_bin = tf.placeholder(tf.float32, (BATCH, SEED_DIM_BIN,))
z_rand_d2 = tf.placeholder(tf.float32, (BATCH, SEED_DIM_D2,))
z_rand = tf.placeholder(tf.float32, (BATCH, SEED_DIM,))
x_data = tf.placeholder(tf.float32, (BATCH, 784))
learning_rate = tf.placeholder(tf.float32, [])

# graph
# --- for D
X_GENERATED = generator(z_rand_bin, z_rand_d2, z_rand)

DISC_REAL = discriminator(x_data)
DISC_FAKE = discriminator(X_GENERATED, reuse=True)
COST_D_raw = tf.reduce_mean(DISC_FAKE) - tf.reduce_mean(DISC_REAL)

VARS_D = [v for v in tf.global_variables() if v.name.startswith("discriminator/")]
vars_g = [v for v in tf.global_variables() if v.name.startswith("generator_conti/")]


eps = tf.random_uniform(
    shape=[BATCH,1],
    minval=0.,
    maxval=1.)

INTERPOLATES = eps * x_data + (1. - eps) * tf.reshape(X_GENERATED, [-1, 784])
GRADIENTS = tf.gradients(discriminator(INTERPOLATES, reuse=True), [INTERPOLATES])[0]
SLOPES = tf.sqrt(tf.reduce_sum(tf.square(GRADIENTS), axis=1))
GRAD_PENALTY = tf.reduce_mean((SLOPES - 1.) ** 2)
COST_D = COST_D_raw + lambd * GRAD_PENALTY

VARS_D = [v for v in tf.global_variables() if v.name.startswith("discriminator/")]
OPTIMIZER_D = tf.train.AdamOptimizer(learning_rate=learning_rate)
GRADS_D = OPTIMIZER_D.compute_gradients(COST_D, var_list=VARS_D)
OP_D = OPTIMIZER_D.apply_gradients(GRADS_D)

# --- for G
x_generated = generator(z_rand_bin, z_rand_d2, z_rand, reuse=True)
disc_real = discriminator(x_data, reuse=True)
disc_fake = discriminator(x_generated, reuse=True)
cost_g = -tf.reduce_mean(disc_fake)

vars_g = [v for v in tf.global_variables() if v.name.startswith("generator/")]
optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_g = optimizer_g.compute_gradients(cost_g, var_list=vars_g)
op_g = optimizer_g.apply_gradients(grads_g)

# --- get samples -----------------------------------------
x_generated_example = generator(z_rand_bin, z_rand_d2, z_rand, reuse=True)

# --- ops to get some tensorboard summaries
def grad_histogram_summaries(grad_var_list):
    l = []
    for grad, var in grad_var_list:
        if grad is not None:
            l += [tf.summary.histogram(var.op.name + '/gradients', grad)]
    return l

def var_histogram_summaries(var_list):
    l = []
    for var in var_list:
        l += [tf.summary.histogram(var.op.name, var)]
    return l


COST_D_SUMM = tf.summary.scalar("COST_D", COST_D)
cost_g_summ = tf.summary.scalar("cost_g", cost_g)

GRADS_D_SUMM = grad_histogram_summaries(GRADS_D)
grads_g_summ = grad_histogram_summaries(grads_g)

VARS_D_SUMM = var_histogram_summaries(VARS_D)
vars_g_summ = var_histogram_summaries(vars_g)

# op for all summaries (but we can also call summaries individually)
MERGED_D_SUMM = tf.summary.merge([COST_D_SUMM, *GRADS_D_SUMM, *VARS_D_SUMM])
merged_g_summ = tf.summary.merge([cost_g_summ, *grads_g_summ, *vars_g_summ])



############### UTILITY FUNCTIONS ################
def get_binary(low=-1., high=1., size=None):
    if size is None:
        size = 1
    l = np.full(size, low)
    h = np.full(size, high)
    rand = np.random.uniform(-1., 1., size)
    bin = np.select([rand<=0., rand>0.], [l, h])
    return bin

def all_binary_combi():
    # generate an nparray of all possible combinations of a vector of 'size' binary values
    return np.array(list(map(list, itertools.product([-1, 1], repeat=SEED_DIM_BIN))))

def get_discrete(low=-1., high=1., num_steps=2, size=None):
    # CHECK !
    if size is None:
        size = 1
    r = np.random.uniform(0., float(num_steps), size)
    # 'size' elements in [0, num_steps]
    r = r.astype(int)
    # size, elements, discretized
    r = r / (num_steps)
    # size, in [0, 1], discretized
    # now scale
    r -= 0.5 #centered
    r *= low - high
    r += (low + high)/2.
    return r

def train_generator(real_images, lr,
                    summary_writer=None, totalstep=None, which_summary=None):

    if summary_writer is None or which_summary is None:
        cost_value, _ = session.run([cost_g, op_g], feed_dict={
            z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
            z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
            z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
            x_data: real_images,
            learning_rate: np.float32(lr),
        })
    else:
        if which_summary=='cost_only':
            summary, cost_value, _ = session.run([cost_g_summ, cost_g, op_g], feed_dict={
                z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
                z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
                z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
                x_data: real_images,
                learning_rate: np.float32(lr),
            })
        elif which_summary=='all':
            summary, cost_value, _ = session.run([merged_g_summ, cost_g, op_g], feed_dict={
                z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
                z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
                z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
                x_data: real_images,
                learning_rate: np.float32(lr),
            })
        summary_writer.add_summary(summary, totalstep)
    return cost_value

def test_generator(real_images):
    cost_value = session.run(cost_g, feed_dict={
        z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
        z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
        z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
        x_data: real_images,
    })
    return cost_value

def train_discriminator(real_images, lr,
                        summary_writer=None, totalstep=None, which_summary=None):

    if summary_writer is None or which_summary is None:
        cost_value, _ = session.run([COST_D, OP_D], feed_dict={
                z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
                z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
                z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
                x_data: real_images,
                learning_rate: np.float32(lr),
            })
    else:
        if which_summary=='cost_only':
            summary, cost_value, _ = session.run([COST_D_SUMM, COST_D, OP_D], feed_dict={
                    z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
                    z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
                    z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
                    x_data: real_images,
                    learning_rate: np.float32(lr),
                })
        elif which_summary=='all':
            summary, cost_value, _ = session.run([MERGED_D_SUMM, COST_D, OP_D], feed_dict={
                    z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
                    z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
                    z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
                    x_data: real_images,
                    learning_rate: np.float32(lr),
                })
        summary_writer.add_summary(summary, totalstep)
    return cost_value

def test_discriminator(real_images):
    cost_value = session.run(COST_D, feed_dict={
        z_rand_bin: get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32),
        z_rand_d2: get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32),
        z_rand: np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32),
        x_data: real_images,
    })
    return cost_value

def example_images(input_z_bin=None, input_z_d2=None, input_z=None):
    if input_z_bin is None:
        input_z_bin= get_binary(size=(BATCH, SEED_DIM_BIN)).astype(np.float32)
    if input_z_d2 is None:
        input_z_d2 = get_discrete(num_steps=4, size=(BATCH, SEED_DIM_D2)).astype(np.float32)
    if input_z is None:
        input_z = np.random.uniform(-1., 1., size=(BATCH, SEED_DIM)).astype(np.float32)
    imgs = session.run(x_generated_example, {
            z_rand_bin: input_z_bin,
            z_rand_d2: input_z_d2,
            z_rand: input_z,
        })[:, :, :, 0]
    return imgs


################### SAVER AND INITIALIZATION #############################""
# saver for all variables
saver = tf.train.Saver()

train_writer = tf.summary.FileWriter('./' + exp_name, session.graph)

# init
if load_from_checkpoint is True:
    if load_model_name is None:
        # get last checkpoint file
        filenames = [filename for filename in os.listdir(exp_name)
                    if filename.startswith('model_totalcsteps')
                    and not filename.endswith('.meta')]
        print(filenames)
        # names are in alphabetical order: 10000 is before 200
        # keep only the longest strings and then select last
        max_len = max([len(filename) for filename in filenames])
        filenames = [filename for filename in filenames if len(filename)==max_len]
        filenames.sort()
        print(filenames)
        filename = filenames[-1]
        print(filename)
    else:
        filename = load_model_name

    completename = exp_name + "/" + filename
    # load variables from it
    saver.restore(session, completename)
    write_to_comment_file("\n" + "loading from checkpointed model " + completename + "\n")
    # restore value of totalcstep
    totaldstep = int(filename[17:])
else:
    session.run(tf.global_variables_initializer())

######################### FOR VIZ #############################"
def check_get_filename(filename, num=None):
    # num: int
    # CHECK if a filename exists, if so return a filename_i version of it
    # in order to not overwrite
    if num is None:
        if os.path.isfile(filename):
            return check_get_filename(filename, 1)
        else:
            return filename
    else:
        if os.path.isfile(filename + "_" + str(num)+".png"):
            return check_get_filename(filename, num+1)
        else:
            return filename + "_" + str(num)

def plot_gen(side, figside=5,
             input_z_bin=None, input_z_d2=None, input_z=None,
             dpi=None, save=None):
    # will plot a square of side*side images
    imgs = example_images(input_z_bin, input_z_d2, input_z)[0:side*side, :, :]
    big_img = np.zeros([28*side, 28*side])
    for i in range(side):
        for j in range(side):
            big_img[i*28:(i+1)*28, j*28:(j+1)*28] = imgs[i*side + j, :, :]
    if save is not None:
        plt.imshow(big_img)
        # check that filename does not already exist. If so, create new filename
        save = check_get_filename(save)
        if dpi is not None:
            plt.savefig(save, dpi=dpi)
        else:
            plt.savefig(save)
    else:
        plt.figure(figsize = (figside, figside))
        plt.imshow(big_img)

############## FOR TESTING ################
def test_discriminator_epoch():
    cost = 0
    for i in range(10000//BATCH):
        batch_x, _ = mnist.test.next_batch(BATCH)
        cost_val = test_discriminator(batch_x)
        cost += cost_val
    cost = cost / (10000//BATCH)
    return cost

def test_generator_epoch():
    cost = 0
    for i in range(10000//BATCH):
        batch_x, _ = mnist.test.next_batch(BATCH)
        cost_val = test_generator(batch_x)
        cost += cost_val
    cost = cost / (10000//BATCH)
    return cost


###################### TRAIN ###########################
# if model is new (not loaded from checkpoint), initialize totaldstep
if load_from_checkpoint is False:
    totaldstep = 0

def train(nbepoch, lr, totaldstep_ini, g_log_ini, d_log_ini, g_log_test_ini, d_log_test_ini):
    nbbatches = nbepoch * 60000 // BATCH
    print("number of G batches to train: " + str(nbbatches))
    totaldstep = totaldstep_ini

    g_costs, g_steps = g_log_ini
    d_costs, d_steps = d_log_ini

    g_costs_test, g_steps_test = g_log_test_ini
    d_costs_test, d_steps_test = d_log_test_ini

    try:
        for i in tqdm(range(nbbatches)):

            ######## TRAIN D ###############################################
            which_g_summary = None
            for s in range(dsteps):
                batch_x, _ = mnist.train.next_batch(BATCH)

                # --- Decide which D summary to save (and transmit to G)
                which_d_summary = None
                if totaldstep % 100 == 0:
                    which_d_summary = "cost_only"
                    if which_g_summary is None:
                        which_g_summary = "cost_only"
                #if totaldstep % 50000 == 0:
                #    which_d_summary = "all"
                #    which_g_summary = "all"

                d_cost = train_discriminator(batch_x, lr,
                                    train_writer, totaldstep, which_d_summary)
                totaldstep += 1
                # occasionally save cost
                if totaldstep % 5 == 0:
                    d_costs += [d_cost]
                    d_steps += [totaldstep]
            #################################################################


            # TRAIN G ################################
            batch_x, _ = mnist.train.next_batch(BATCH)
            g_cost = train_generator(batch_x, lr,
                            train_writer, totaldstep,
                            which_g_summary)
            # occasionally save cost
            if totaldstep % 5 == 0:
                g_costs += [g_cost]
                g_steps += [totaldstep]
            ##########################################

            # occasionally test performance on test set
            if i % 50 == 0:
                g_costs_test += [test_generator_epoch()]
                g_steps_test += [totaldstep]
                d_costs_test += [test_discriminator_epoch()]
                d_steps_test += [totaldstep]

            # occasionally save some generated images
            if i%5000==0:
                plot_gen(side=10, figside=12, dpi=200, save=exp_name + "/img_" + str(totaldstep)+".png")

    except KeyboardInterrupt:
        print("Interrupted")

    g_log = g_costs, g_steps
    d_log = d_costs, d_steps
    g_log_test = g_costs_test, g_steps_test
    d_log_test = d_costs_test, d_steps_test

    return totaldstep, g_log, d_log, g_log_test, d_log_test

g_log = [], []
d_log = [], []
g_log_test = [], []
d_log_test = [], []

totaldstep, g_log, d_log, g_log_test, d_log_test = train(train_epochs, 1e-3, totaldstep, g_log, d_log, g_log_test, d_log_test)

np.savez(exp_name + "/g_log", costs=np.array(g_log[0]), steps=np.array(g_log[1]))
np.savez(exp_name + "/d_log", costs=np.array(d_log[0]), steps=np.array(d_log[1]))
np.savez(exp_name + "/g_log_test", costs=np.array(g_log_test[0]), steps=np.array(g_log_test[1]))
np.savez(exp_name + "/d_log_test", costs=np.array(d_log_test[0]), steps=np.array(g_log_test[1]))

plt.gcf().clear()
plt.plot(g_log[1], g_log[0], label='train')
plt.plot(g_log_test[1], g_log_test[0], label='test')
plt.legend()
plt.xlabel('totaldsteps')
plt.ylabel('g_cost')
plt.xlim(xmin=500)
plt.savefig(exp_name + '/g_costs.png')

plt.gcf().clear()
plt.plot(d_log[1], d_log[0], label='train')
plt.plot(d_log_test[1], d_log_test[0], label='test')
plt.legend()
plt.xlabel('totaldsteps')
plt.ylabel('d_cost')
plt.xlim(xmin=500)
plt.savefig(exp_name + '/d_costs.png')

plt.gcf().clear()
plot_gen(side=10, save=exp_name + '/generated.png')

