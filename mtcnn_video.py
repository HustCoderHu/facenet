from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
# print(os.getcwd())
import argparse
import tensorflow as tf
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "\\..")
# ModuleNotFoundError: No module named 'facenet'
import facenet
import align.detect_face
import random
from time import sleep
import cv2 as cv
import json

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def main(args):
  # sleep(random.random())
  output_dir = os.path.expanduser(args.output_dir)
  # Store some git revision info in a text file in the log directory
  # src_path, _ = os.path.split(os.path.realpath(__file__))
  # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

  minsize = 20  # minimum size of face
  threshold = [0.9, 0.9, 0.9]  # three steps's threshold
  # threshold = [0.7, 0.6, 0.8]  # three steps's threshold
  # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
  factor = 0.709  # scale factor

  input_dir = args.input_dir
  image_name = "tcm_0.png"
  video_name = "the_chinaman.mp4"
  boxes_file = "video_boxes.json"
  output_video = "drawn.avi"
  # image_name = "lab_0004.jpg"

  image_path = os.path.join(input_dir, image_name)
  video_path = os.path.join(input_dir, video_name)

  boxes_json_path = os.path.join(output_dir, boxes_file)
  output_video_path = os.path.join(output_dir, output_video)

  #
  draw_boxes_on_video(video_path, output_video_path, boxes_json_path)
  return

  # print(image_path)
  # print(os.path.exists(image_path))


  # output_class_dir = os.path.join(output_dir, image_name)
  # if not os.path.exists(output_class_dir):
    # os.makedirs(output_class_dir)
  winname = "det"
  # cv.namedWindow(winname, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
  batch_size = 50
  cap = cv.VideoCapture(video_path)
  frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
  fps = cap.get(cv.CAP_PROP_FPS)

  # print(type(frame_count))
  print(frame_count)
  # print(fps)
  # cap.release()
  # return

  frame_count = int(frame_count)
  video_boxes = []
  for _ in range(frame_count):
    video_boxes.append([]) # box list per frame


  start_frame = fps * 60
  end_frame = fps * 80
  frame_count = 0
  while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
      break
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    print(frame_count)
    if frame_count <start_frame:
      frame_count = frame_count + 1
      continue
    if frame_count >= end_frame:
      break
    # start detection
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    # detect faces
    if nrof_faces > 0:
      det = bounding_boxes[:, 0:4]
      det_arr = []
      img_size = np.asarray(img.shape)[0:2]
      for i in range(nrof_faces):
        det_arr.append(np.squeeze(det[i]))

      boxes = []
      for i, det in enumerate(det_arr):
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)

        args.margin = 22
        left = np.maximum(det[0] - args.margin / 2, 0)
        top = np.maximum(det[1] - args.margin / 2, 0)
        right = np.minimum(det[2] + args.margin / 2, img_size[1])
        bottom = np.minimum(det[3] + args.margin / 2, img_size[0])
        # bb[0] = np.maximum(det[0] - args.margin / 2, 0)
        # bb[1] = np.maximum(det[1] - args.margin / 2, 0)
        # bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
        # bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
        boxes.append([left, top, right, bottom])
      video_boxes[frame_count] = boxes

    frame_count = frame_count + 1

  print("write to file: {}".format(boxes_json_path))
  boxes_json = {video_name:video_boxes}
  with open(boxes_json_path, "w") as f:
    json.dump(boxes_json, f)
  print("finished")

    # frame_list.append(frame)
    # cv.imshow(winname, frame)
    # if cv.waitKey(30) & 0xFF == ord('q'):
    #   break
  # print(len(frame_list))
  # print(frame_count)
  cap.release()
  cv.destroyAllWindows()
  return

  img = misc.imread(image_path)
  bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  nrof_faces = bounding_boxes.shape[0]
  # detect faces
  if nrof_faces > 0:
    det = bounding_boxes[:, 0:4]
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    for i in range(nrof_faces):
      det_arr.append(np.squeeze(det[i]))
    for i, det in enumerate(det_arr):
      det = np.squeeze(det)
      bb = np.zeros(4, dtype=np.int32)

      args.margin = 22
      bb[0] = np.maximum(det[0] - args.margin / 2, 0)
      bb[1] = np.maximum(det[1] - args.margin / 2, 0)
      bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
      bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
      boxes.append(bb)

  img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
  # img = cv.imread(image_path)
  # height = img.shape[0]
  # width = img.shape[1]
  for bb in boxes:
    # bb = [width - bb[0], height - bb[1], width - bb[2], height - bb[3]]
    cv.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0,255,0), 1)
  winname = "det"
  cv.namedWindow(winname, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
  # print(img)
  # print(type(img)) # ndarray
  # print(img.shape)
  # cv.resizeWindow(winname, tuple([shape for shape in img.shape]))
  # cv.resizeWindow(winname, [size for size in img.shape])
  # cv.resizeWindow(winname, img.shape[0]/3, img.shape[1]/3)
  cv.imshow(winname, img)
  cv.waitKey()
  # img = img[:, :, 0:3]

def draw_boxes_on_video(in_video, out_video, boxes_json):
  boxes_dict = {}
  with open(boxes_json, "r") as f:
    boxes_dict = json.load(f)

  video_boxes = []
  for _video_boxes in boxes_dict.values():
    print(type(video_boxes))
    print(len(video_boxes))
    video_boxes = _video_boxes

  cap = cv.VideoCapture(in_video)
  fps = int(cap.get(cv.CAP_PROP_FPS))
  height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  print(height)
  print(width)

  # Define the codec and create VideoWriter object
  # fourcc = cv.VideoWriter_fourcc(*'DIVX')
  fourcc = cv.VideoWriter_fourcc(*'XVID')
  out = cv.VideoWriter(out_video, fourcc, fps, (width, height))
  # out = cv.VideoWriter(out_video, fourcc, fps, (height, width))

  start_frame = fps * 60
  end_frame = fps * 80

  cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
  for i in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
      break
    print("frame: {}".format(i))
    for box in video_boxes[i]:
      left_top = tuple(int(pos) for pos in box[:2])
      # print(type(left_top))
      right_bottom = tuple(int(pos) for pos in box[2:])
      cv.rectangle(frame, left_top, right_bottom, (0, 255, 0), 3)
      out.write(frame)
      # cv.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)

  # Release everything if job is finished
  cap.release()
  out.release()
  return


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
  parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
  parser.add_argument('--image_size', type=int,
                      help='Image size (height, width) in pixels.', default=182)
  parser.add_argument('--margin', type=int,
                      help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
  parser.add_argument('--random_order',
                      help='Shuffles the order of images to enable alignment using multiple processes.',
                      action='store_true')
  parser.add_argument('--gpu_memory_fraction', type=float,
                      help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
  parser.add_argument('--detect_multiple_faces', type=bool,
                      help='Detect and align multiple faces per image.', default=True)
  return parser.parse_args(argv)


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))