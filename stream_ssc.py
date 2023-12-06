import subprocess
import time
import datetime
from concurrent.futures import ThreadPoolExecutor


def get_stream_url(url):
    try:
        result = subprocess.run(["youtube-dl", "-g", url], stdout=subprocess.PIPE)
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"Error fetching stream URL: {e}")
        return None


def pull_frames(stream_url, name):
    try:
        # Get the current timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Use the name and timestamp in the filename
        filename = f"stream_inputs/{name}_{current_time}.png"

        subprocess.run(
            ["ffmpeg", "-i", stream_url, "-vf", "fps=1", "-t", "1", filename]
        )
    except Exception as e:
        print(f"Error downloading frames: {e}")


def process_video(url, name):
    stream_url = get_stream_url(url)
    if stream_url:
        pull_frames(stream_url, name)


if __name__ == "__main__":
    URL_Name_List = [
        ("https://www.youtube.com/live/gCNeDWCI0vo?si=AAxtZcpFBL26CbEG", "Al_Jazeera"),
        ("https://www.youtube.com/watch?v=tkDUSYHoKxE", "France_24"),
    ]

    while True:
        with ThreadPoolExecutor() as executor:
            # Process each URL concurrently
            executor.map(lambda args: process_video(*args), URL_Name_List)

        # time.sleep(1)  # Sleep for 1 second before processing URLs again
