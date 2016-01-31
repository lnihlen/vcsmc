#!/usr/bin/env python

import argparse
import glob
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='foo blah blah')
parser.add_argument('--frame_csv', required=True)
parser.add_argument('--stills_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--audio_dir', required=True)
parser.add_argument('--gen_path', default='out/gen')
parser.add_argument('--quality', default='45')

# Returns list of tuples (is_keyframe (bool), presentation_time_s (double))
def parse_csv_file(frame_csv):
  csv = open(frame_csv)
  if not csv:
    print 'error opening frame_csv file ' + frame_csv
    sys.exit(-1)
  frames = []
  for line in csv.readlines():
    s = line.split(',')
    if s[0] != 'frame' or s[1] != 'video':
      print 'skipping odd line: ' + line
    is_keyframe = s[2] == '1'
    t = float(s[5])
    frames.append((is_keyframe, t))
  return frames

def make_output_directory_structure(root_dir):
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  if not os.path.exists(os.path.join(root_dir, 'generations')):
    os.makedirs(os.path.join(root_dir, 'generations'))
  if not os.path.exists(os.path.join(root_dir, 'stills')):
    os.makedirs(os.path.join(root_dir, 'stills'))
  if not os.path.exists(os.path.join(root_dir, 'kernels')):
    os.makedirs(os.path.join(root_dir, 'kernels'))
  if not os.path.exists(os.path.join(root_dir, 'ideal')):
    os.makedirs(os.path.join(root_dir, 'ideal'))

def main(args):
  frames = parse_csv_file(args.frame_csv)
  make_output_directory_structure(args.output_dir)
  stills = glob.glob(os.path.join(args.stills_dir, '*.png'))
  if len(frames) != len(stills):
    print 'frame data %d/image file count %d mismatch' \
        % (len(frames), len(stills))
    sys.exit(-1)
  audio_frames = glob.glob(os.path.join(args.audio_dir, '*.spec'))
  quality = int(args.quality)
  current_frame = 0
  current_still = 0
  while current_frame < len(audio_frames):
    print 'frame %d of %d, still %d of %d' \
            % (current_frame + 1, \
               len(audio_frames), \
               current_still + 1, \
               len(frames))
    # See if we need to advance image.
    frame_time = float(current_frame) * (1.0 / 60.0)
    reset_image = current_frame == 0
    if current_still < len(frames) - 1 and \
        frame_time >= frames[current_still + 1][1]:
      current_still += 1
      # We only don't seed from previous image generation file if the new frame
      # is a keyframe.
      if frames[current_still][0]:
        reset_image = True
    gen_output_file = os.path.join(args.output_dir, 'generations', \
        'frame-%07d.yaml' % (current_frame))
    image_output_file = os.path.join(args.output_dir, 'stills', \
        'frame-%07d.png' % (current_frame))
    kernel_output_file = os.path.join(args.output_dir, 'kernels', \
        'frame-%07d.yaml' % (current_frame))
    ideal_output_file = os.path.join(args.output_dir, 'ideal', \
        'frame-%07d.png' % (current_frame))
    command_line = \
            [args.gen_path,
            '--target_percent_error=%d' % (quality),
            '--target_image_file=%s' % (stills[current_still]),
            '--generation_output_file=%s' % (gen_output_file),
            '--image_output_file=%s' % (image_output_file),
            '--global_minimum_output_file=%s' % (kernel_output_file),
            '--audio_spec_list_file=%s' % (audio_frames[current_frame]),
            '--ideal_image_output_file=%s' % (ideal_output_file)]
    # Seeding on previous generation. We always seed on an existing generation
    # file if there is one.
    if os.path.exists(gen_output_file):
      command_line.append('--seed_generation_file=%s' % (gen_output_file))
    elif not reset_image:
      command_line.append('--seed_generation_file=%s' % \
          os.path.join(args.output_dir, 'generations', 'frame-%07d.yaml' % \
          (current_frame - 1)))
    exit_code = subprocess.call(command_line)
    if exit_code != 0:
      print 'got error, exiting.'
      sys.exit(exit_code)
    current_frame += 1

  return 0

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
