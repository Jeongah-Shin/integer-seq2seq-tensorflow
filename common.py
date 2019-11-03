import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def create_folder(location):
    if not os.path.exists(location):
        os.makedirs(location)

def get_session_config():
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    return config

def get_checkpoint(checkpoint_dir, log=True):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if log:
        tf.logging.info("loading tensorflow checkpoints...")
        tf.logging.info('[+] Get checkpoint {}'.format(ckpt))

    if ckpt and ckpt.model_checkpoint_path:
        last_step = int(ckpt.model_checkpoint_path.split("-")[-1])
        ckpt_path = ckpt.model_checkpoint_path
        if log:
            tf.logging.info("[+] RESTORE SAVED VARIBALES : restored {}".format(ckpt_path))
            tf.logging.info("[+] RESTORE SAVED VARIBALES : restart from step {}".format(last_step))
    else:
        raise RuntimeError('checkpoint file was not found')
    return ckpt_path, last_step


def get_batch_nums(batch_size, gpus):
    q, r = divmod(batch_size, gpus)
    return [q + 1] * r + [q] * (gpus - r)

def get_init_pretrained():
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )

    init_fn = lambda sess, ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return list(map(lambda x: os.path.join(img_dir,x), imgs)),\
            list(map(lambda x: os.path.join(img_dir,x), masks)),\
            list(map(lambda x: os.path.join(img_dir,x), xmls))

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
            else:
                print ('There is no file : %s'%(file))
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


