from xeuclid.euclid2 import *
from xeuclid.tikz_draw import *
from xeuclid.utils.file_edit import *

class Anim():

    def __init__(self, folder_name, duration, fps=23):
        self.folder_name = folder_name
        self.__frame = 0
        self.fps = fps
        self.duration = duration
        self.frames = self.fps * self.duration

    @property
    def frame(self):
        return self.__frame

    def setup(self):
        pass

    def update(self):
        pass

    def create_frames(self):
        self.setup()
        for i in range(1, self.frames + 1):
            self.tikz = Tikz("__tex__.tex", preamble=standalone)
            self.update()
            self.tikz.png(out_path=self.folder_name+"\\")
            os.rename(f"{self.folder_name}\\page0001-1.png", f"{self.folder_name}\\"+((3 - len(str(i)))*"0")+str(i)+".png")
            os.system("del __tex__.tex __tex__.pdf")
            self.__frame += 1

    def compile_vid(self):
        self.create_frames()
        os.system(f"cd {self.folder_name} && ffmpeg -i %03d.png -vf scale=1080:1080 %03d.png")
        os.system(f"cd {self.folder_name} && ffmpeg -framerate {self.fps} -i %03d.png -pix_fmt yuv420p movie.mp4")

def compile_anim(anim):
    anim.compile_vid()