import os, re, os.path
import os
from xeuclid.utils.file_edit import File
from xeuclid import __version__

def clear_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))

if 'dist' in os.listdir():
    clear_dir('dist')

if 'xeuclid.egg-info' in os.listdir():
    clear_dir('xeuclid.egg-info')

cfg = File("setup.cfg")
cfg.replace(f"version = {__version__}\n", 3)

os.system('py -m build')
os.system('twine upload dist/*')
