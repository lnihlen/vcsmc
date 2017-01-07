#!/usr/bin/env python

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description='make a movie for vcsmc')
parser.add_argument('--audio_dir', required=True)
parser.add_argument('--debug', default=False)
parser.add_argument('--fit_path', default='out/fit')
parser.add_argument('--frame_csv', required=True)
parser.add_argument('--gen_path', default='out/gen')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--stagnant_limit', default='5')
parser.add_argument('--stills_dir', required=True)
parser.add_argument('--verbose', default=False)
parser.add_argument('--start_at_frame', default=0)

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
  stills = sorted(glob.glob(os.path.join(args.stills_dir, '*.png')))
  if len(frames) != len(stills):
    print 'frame data %d/image file count %d mismatch' \
        % (len(frames), len(stills))
    sys.exit(-1)
  audio_frames = sorted(glob.glob(os.path.join(args.audio_dir, '*.spec')))
  current_frame = 0
  current_still = 0
  current_hash = None
  start_at_frame = int(args.start_at_frame)
  # Path to final output binary.
  movie_binary = os.path.join(args.output_dir, 'vcsmc.bin')
  while current_frame < len(audio_frames):
    print 'frame %d of %d, still %d of %d' \
            % (current_frame, \
               len(audio_frames), \
               current_still, \
               len(frames))
    # See if we need to advance image.
    frame_time = float(current_frame) * (1.0 / 60.0)
    if current_still < len(frames) - 1 and \
        frame_time >= frames[current_still + 1][1]:
      current_still += 1
      current_hash = None
    if current_frame < start_at_frame:
      if args.verbose:
        print 'current_frame %d less than starting frame %d, skipping.' \
            % (current_frame, start_at_frame)
      current_frame += 1
      continue
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
        'frame-%07d.yaml' % (current_frame))
    fit_color_file = os.path.join(args.output_dir, 'fit', \
        current_hash + '.col')
    image_output_file = os.path.join(args.output_dir, 'stills', \
        'frame-%07d.png' % (current_frame))
    command_line = \
        [args.gen_path,
        '--color_input_file=%s' % (fit_color_file),
        '--image_output_file=%s' % (image_output_file),
        '--global_minimum_output_file=%s' % (kernel_file),
        '--audio_spec_list_file=%s' % (audio_frames[current_frame]),
        '--target_error=0.0',
        '--stagnant_count_limit=%s' % (args.stagnant_limit),
        '--append_kernel_binary=%s' % (movie_binary)]
    if args.debug:
      command_line = ['lldb', '--batch', '--one-line', 'run', '--file'] + \
          command_line[0:1] + ['--'] + command_line[1:]
    if args.verbose:
      command_line.append('--print_stats=true')
      print command_line
    try:
      subprocess.call(command_line)
    except KeyboardInterrupt:
      if args.debug:
        time.sleep(3600)
      else:
        raise
    current_frame += 1

  return 0

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
