#!/usr/bin/env python

import argparse
import glob
import os
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description='make a movie for vcsmc')
parser.add_argument('--audio_dir', required=True)
parser.add_argument('--fit_path', default='out/fit')
parser.add_argument('--frame_csv', required=True)
parser.add_argument('--gen_path', default='out/gen')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--respec_path', default='out/respec')
parser.add_argument('--stagnant_limit', default=5)
parser.add_argument('--stills_dir', required=True)
parser.add_argument('--verbose', default=True)

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
  if not os.path.exists(os.path.join(root_dir, 'fit')):
    os.makedirs(os.path.join(root_dir, 'fit'))
  if not os.path.exists(os.path.join(root_dir, 'stills')):
    os.makedirs(os.path.join(root_dir, 'stills'))
  if not os.path.exists(os.path.join(root_dir, 'kernels')):
    os.makedirs(os.path.join(root_dir, 'kernels'))

def main(args):
  frames = parse_csv_file(args.frame_csv)
  make_output_directory_structure(args.output_dir)
  stills = glob.glob(os.path.join(args.stills_dir, '*.png'))
  if len(frames) != len(stills):
    print 'frame data %d/image file count %d mismatch' \
        % (len(frames), len(stills))
    sys.exit(-1)
  audio_frames = glob.glob(os.path.join(args.audio_dir, '*.spec'))
  current_frame = 0
  current_still = 0
  current_hash = None
  # Path to final output binary.
  movie_binary = os.path.join(args.output_dir, 'vcsmc.bin')
  while current_frame < len(audio_frames):
    print 'frame %d of %d, still %d of %d' \
            % (current_frame + 1, \
               len(audio_frames), \
               current_still + 1, \
               len(frames))
    # See if we need to advance image.
    frame_time = float(current_frame) * (1.0 / 60.0)
    if current_still < len(frames) - 1 and \
        frame_time >= frames[current_still + 1][1]:
      current_still += 1
      current_hash = None
    # Compute hash of frame if currently unknown.
    if not current_hash:
      command_line = \
          [args.fit_path,
          '--image_file=%s' % (stills[current_still]),
          '--output_dir=%s' % (os.path.join(args.output_dir, 'fit'))]
      if args.verbose:
        print command_line
      current_hash = subprocess.check_output(command_line)[:-1]
    kernel_file = os.path.join(args.output_dir, 'kernels', \
        current_hash + '.yaml')
    fit_color_file = os.path.join(args.output_dir, 'fit', \
        current_hash + '.col')
    fit_image_file = os.path.join(args.output_dir, 'kernels', \
        current_hash + '.png')
    if os.path.exists(kernel_file):
      command_line = \
          [args.respec_path,
          '--input_kernel_file=%s' % (kernel_file),
          '--audio_spec_file=%s' % (audio_frames[current_frame]),
          '--append_kernel_binary=%s' % (movie_binary)]
      if args.verbose:
        print command_line
      subprocess.call(command_line)
    else:
      command_line = \
          [args.gen_path,
          '--color_input_file=%s' % (fit_color_file),
          '--image_output_file=%s' % (fit_image_file),
          '--global_minimum_output_file=%s' % (kernel_file),
          '--audio_spec_list_file=%s' % (audio_frames[current_frame]),
          '--target_error=0.0',
          '--stagnant_count_limit=%d' % (args.stagnant_limit),
          '--append_kernel_binary=%s' % (movie_binary)]
      if args.verbose:
        command_line.append('--print_stats=true')
        print command_line
      subprocess.call(command_line)
    image_output_file = os.path.join(args.output_dir, 'stills', \
        'frame-%07d.png' % (current_frame))
    # Copy fit image into place in frames.
    if args.verbose:
      print 'copying %s to %s' % (fit_image_file, image_output_file)
    shutil.copy(fit_image_file, image_output_file)
    current_frame += 1

  return 0

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
