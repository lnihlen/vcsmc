import os
import sys

from wand.image import Image

def main(argv):
  input_file_spec = ''
  output_file_path = ''
  if len(argv) != 3:
    print 'usage: proeprocess_images.py <input_file_spec> <output_path>'
    sys.exit(-1)

  input_file_spec = argv[1]
  output_file_path = argv[2]

  file_number = 1
  file_path = input_file_spec % file_number
  while os.path.exists(file_path):
    print 'processing %s..' % file_path
    img = Image(filename=file_path)
    # remove any letterboxing from top and bottom
    img.trim()
    # resize to 180px tall at original aspect ratio
    original_size = img.size()
    scale = 180.0 / float(original_size[1])
    scale_width = int(scale * float(original_size[0]))
    img.resize(width=scale_width, height=180)

if __name__ == '__main__':
  main(sys.argv)
