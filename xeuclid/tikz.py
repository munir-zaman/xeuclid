import subprocess
import os
from xeuclid.euclid2 import *
from xeuclid.utils.file import *

# trying to create a more polished version of tikz_draw.py

template = \
"""\\documentclass[tikz, border=5, svgnames]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}"""

packages = ['xcolor', 'amsmath', 'amsfonts', 'amssymb']
libs = ['arrows.meta']

class Tikz:
    def __init__(self, fp, template=template, packages=packages, tikzlibs=libs):
        self.fp = fp
        self.template = template
        self.packages = packages
        self.tikzlibs = tikzlibs

    def add_packages(self, *packages):
        self.packages = self.packages + list(packages)

    def add_tikzlibs(self, *libs):
        self.tikzlibs = self.tikzlibs + list(libs)

    def create_file(self):
        self.file = File(self.fp)
        self.file.create()

    def write(self, text, end="\n"):
        self.file.write(text, end=end)

    def init(self):
        # write template
        # load packages
        # load tikzlibs

        self.create_file() # create the file
        self.write(template) # write the template

        usepackages = [f"\\usepackage{{{package}}}" for package in self.packages]
        self.write(usepackages) # add packages

        usetikzlibs = [f"\\usetikzlibrary{{{lib}}}" for lib in self.tikzlibs]
        self.write(usetikzlibs) # add tikzlibs

        self.write("") # empty line

