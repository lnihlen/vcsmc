import glob
import os
import struct
import subprocess
import sys

def main(argv):
  # make an out directory if it does not exist
  if not os.path.exists('out') or not os.path.isdir('out'):
    os.makedirs('out')

  # glob all _sim.v files
  sims = sorted(glob.glob('*_sim.v'))
  for sim_name in sims:
    name = sim_name[:-6]
    out_name = os.path.join('out', sim_name[:-2])
    build_output = subprocess.check_output(['iverilog', '-Wall', '-o', out_name, sim_name])
    if build_output:
      print 'error compiling %s:\n%s' % (sim_name, build_output)
      continue
    sim_output = subprocess.check_output(['vvp', out_name])
    expected_out = 'VCD info: dumpfile out/%s opened for output.\nOK\n' % (sim_name[:-2] + '.vcd')
    if sim_output != expected_out:
      print '%s:\n%s\n' % (sim_name, sim_output)

if __name__ == "__main__":
  main(sys.argv)
