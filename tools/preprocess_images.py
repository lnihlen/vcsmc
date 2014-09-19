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
    output_root_file = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_file_path, output_root_file + '.tiff')
    print 'processing %s to %s' % (file_path, output_path)
    img = Image(filename=file_path)
    # remove any letterboxing from top and bottom, unfortunately this eats other
    # elements too which can end up interfering negatively with cropping
#    trimmer = img.clone()
#    trimmer.trim()
#    if trimmer.size[0] > 1:
#      img = trimmer
#      img.reset_coords()
    # resize to 180px tall at original aspect ratio
    original_size = img.size
    scale = 192.0 / float(original_size[1])
    scale_width = int(scale * float(original_size[0]))
    # always make even width for centering, etc.
#    if scale_width % 2 == 1:
#      scale_width -= 1
    img.resize(width=scale_width, height=192)
    img.save(filename=output_path)
    file_number += 1
    file_path = input_file_spec % file_number

if __name__ == '__main__':
  main(sys.argv)
