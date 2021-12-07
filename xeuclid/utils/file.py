import os
import shutil

class File:
    def __init__(self, fp):
        self.fp = fp

    def create(self):
        if os.path.exists(self.fp):
            os.remove(self.fp)
            # delete the file if it exists

        with open(self.fp, 'x') as file:
            pass
            # create the file

    def write(self, lines, end="\n"):
        if isinstance(lines, str):
            lines+= "\n"
        elif isinstance(lines, (list, tuple)):
            lines = list(lines)
            lines = [line+"\n" for line in lines]
        # add ``end`` to every line

        with open(self.fp, 'a') as file:
            file.writelines(lines)

    def clear(self):
        with open(self.fp, 'w') as file:
            pass

    def read(self):
        with open(self.fp, 'r') as file:
            file_text = file.read()
        return file_text

    def copy(self, dst):
        shutil.copyfile(self.fp, dst)

    def move(self, dst):
        shutil.move(self.fp, dst)

    def rename(self, dst):
        os.rename(self.fp, dst)
        self.fp = dst

    def remove(self):
        os.remove(self.fp)
        self.fp = None

