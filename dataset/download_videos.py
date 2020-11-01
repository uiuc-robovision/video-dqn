from pytube import YouTube, exceptions
# import cv2
from os import listdir, mkdir, rename, system
import numpy as np
from tqdm import tqdm
import util
import urllib
import time
#import matplotlib.pyplot as plt
#import pandas as pd
# %matplotlib inline

import subprocess


class NoStreamFound(Exception):
    pass


urls = np.load('urls.npy')
url_prefix = 'https://www.youtube.com/watch?v='

print('Num videos:', urls.shape[0])
failures = []

try:
    mkdir('videos')
    print('creating videos subdirectory')
except:
    print('videos subdirectory already exists')

num_failed = 0
videos = util.files('./videos')
sleep_time = 10

# TODO
completed = []
remaining = set(urls) - set(completed)
print("Num Remaining: ",len(remaining))
for vid_id in remaining:
    while True:
        print('starting', vid_id)

        url = url_prefix + vid_id

        try:
            print('url: ', url)
            video = YouTube(url)
            v1080 = [
                e for e in video.streams.filter(file_extension='mp4')
                if e.resolution == '1080p'
            ]
            v720 = [
                e for e in video.streams.filter(file_extension='mp4')
                if e.resolution == '720p'
            ]
            streams = v1080 + v720
            if len(streams) == 0:
                raise NoStreamFound()

            highResVideo = streams[0]
            file_loc = highResVideo.download('videos')
            rename(file_loc, 'videos/' + str(vid_id) + '.mp4')
        except (exceptions.VideoUnavailable, exceptions.RegexMatchError,
                NoStreamFound, urllib.error.HTTPError) as e:
            print(e)
            # too many requests error, exponential backoff 
            if type(e) == urllib.error.HTTPError and e.code == 429:
                print("backoff")
                time.sleep(sleep_time)
                sleep_time *= 2
                continue
            print("Failed on ", vid_id)
            failures.append(vid_id)
        sleep_time = 10
        break;
print("failures:", failures)
