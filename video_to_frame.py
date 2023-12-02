import os

bashCommand = "ffmpeg -i 5secbig.mp4 -vf format=gray -r 1/1 imgtest/%03d.jpg"
os.system(bashCommand)
