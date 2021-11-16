import os

import cv2
import moviepy.video.io.ImageSequenceClip


def split_video_to_frames(path_to_video: str) -> None:
    cam = cv2.VideoCapture(path_to_video)
    name, _ = os.path.splitext(os.path.basename(path_to_video))
    path_to_save = os.path.join(os.path.dirname(path_to_video), name + '_' + 'data')
    try:
        # creating a folder named data
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0
    while True:
        # reading from frame
        ret, frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = os.path.join(path_to_save, 'frame_' + str(currentframe) + '.jpg')
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def convert_frames_to_video(images_dir: str, path_out: str, fps=15) -> None:
    image_files = [os.path.join(images_dir, img)
                   for img in os.listdir(images_dir)]
    image_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(path_out, codec="libx264")


if __name__ == '__main__':
    # split_video_to_frames('examples/video.mp4')
    convert_frames_to_video('examples/crooped_video_data', 'videos/my_favorite_gif.MP4', fps=6)
