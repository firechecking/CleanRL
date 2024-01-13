# -*- coding: utf-8 -*-
# @Time    : 2024/1/13 14:31
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : video_writer.py
# @Software: CleanRL
# @Description: video_writer

import copy
import pyglet
from moviepy.editor import ImageSequenceClip


class PreviewWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(width=420, height=420, vsync=False, resizable=True, *args, **kwargs)

    def show(self, frame):
        self.clear()
        self.switch_to()
        self.dispatch_events()

        image = pyglet.image.ImageData(
            frame.shape[1],
            frame.shape[0],
            'RGB',
            frame.tobytes(),
            pitch=frame.shape[1] * -3
        )
        image.blit(0, 0, width=self.width, height=self.height)
        self.flip()


class VideoWriter():
    def __init__(self):
        self.window = PreviewWindow()
        self.frames = []

    def add_frame(self, frame):
        self.window.show(frame)
        self.frames.append(copy.deepcopy(frame))

    def flush(self, out_fn, fps=10):
        if out_fn:
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(out_fn, fps=fps)
        self.window.close()


if __name__ == "__main__":
    pass
