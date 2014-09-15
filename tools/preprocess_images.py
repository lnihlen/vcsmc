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
    # remove any letterboxing from top and bottom
    trimmer = img.clone()
    trimmer.trim()
    if trimmer.size[0] > 1:
      img = trimmer
      img.reset_coords()
    # resize to 180px tall at original aspect ratio
    original_size = img.size
    scale = 180.0 / float(original_size[1])
    scale_width = int(scale * float(original_size[0]))
    img.resize(width=scale_width, height=180)
    if scale_width > 320:
      left_crop = (scale_width - 320) / 2
      img = img[left_crop:left_crop + 320, 0:180]

    img.save(filename=output_path)
    file_number += 1
    file_path = input_file_spec % file_number

if __name__ == '__main__':
  main(sys.argv)
