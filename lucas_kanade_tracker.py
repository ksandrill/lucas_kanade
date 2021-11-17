import os

import cv2
import numpy as np

from util import convolve, normalize_image, draw_arrow, create_color_gray

PATH_TO_DUMPTRUCK = 'examples/video_data'
PATH_TO_SAVE = 'examples/crooped_video_data'


def get_dX(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, 1, 0], [-1, 1, 0], [0, 0, 0]])
    dx = convolve(image, kernel)
    return dx


def get_dY(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, 0], [1, 1, 0], [0, 0, 0]])
    dy = convolve(image, kernel)
    return dy


def get_dt(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    dt = convolve(image1, kernel) + convolve(image2, -kernel)
    return dt


def find_interesting_dots(image_dX: np.ndarray, image_dY: np.ndarray, thr: float, const: float, window_size: int = 3) \
        -> list[(int, int)]:
    dots = []
    h, w = image_dX.shape
    square_dX = image_dX ** 2
    square_dY = image_dY ** 2
    dX_dY = image_dX * image_dY
    offset = window_size // 2
    for i in range(offset, h - offset):
        start_i = i - offset
        end_i = i + offset + 1
        for j in range(offset, w - offset):
            start_j = j - offset
            end_j = j + offset + 1
            ss_xx = np.sum(square_dX[start_i:end_i, start_j:end_j])
            ss_yy = np.sum(square_dY[start_i:end_i, start_j:end_j])
            ss_xy = np.sum(dX_dY[start_i:end_i, start_j:end_j])
            det = ss_xx * ss_yy - ss_xy ** 2
            trace = ss_xx + ss_yy
            response = det - const * trace ** 2
            if response >= thr:
                dots.append((j, i))  # x y
    return dots


def calc_interesting_point_with_velocity(first_image: np.ndarray, second_image: np.ndarray, harris_thr: float = 0.1,
                                         harris_constant: float = 0.06) -> list[((int, int), (float, float))]:
    first_dX = get_dX(first_image)
    first_dY = get_dY(first_image)
    dots = find_interesting_dots(first_dX, first_dY, thr=harris_thr, const=harris_constant)
    dT = get_dt(first_image, second_image)
    offset = 1  # 3x3 window_size
    points_with_velocity = []
    for (j, i) in dots:
        start_i = i - offset
        end_i = i + offset + 1
        start_j = j - offset
        end_j = j + offset + 1
        Ix = first_dX[start_i:end_i, start_j:end_j].flatten()
        Iy = first_dX[start_i:end_i, start_j:end_j].flatten()
        It = dT[start_i:end_i, start_j:end_j].flatten()
        A = np.array([Ix, Iy])
        A_trans = np.array(A)  # transpose of A
        A = np.array(np.transpose(A))
        A_pinv = np.linalg.pinv(np.dot(A_trans, A))
        vel_x, vel_y = np.dot(np.dot(A_pinv, A_trans), It)  # Velocity = (A_Transpose  @ A) @ A_Transpose  @ It
        # print(vel_x, vel_y)
        points_with_velocity.append(((j, i), (vel_x, vel_y)))
    return points_with_velocity


def visual_threshold_moving_points(color_gray: np.ndarray, points_with_velocity: list[((int, int), (float, float))],
                                   thr: float = 0.01) -> None:
    for ((j, i), (vel_x, vel_y)) in points_with_velocity:
        if abs(vel_x) >= thr and abs(vel_y) >= thr:
            draw_arrow(color_gray, (i, j), (int(i + 3 * vel_x), int(j + 3 * vel_y)), (0, 0, 1))


def main():
    path_to_images = [os.path.join(PATH_TO_DUMPTRUCK, item) for item in os.listdir(PATH_TO_DUMPTRUCK)]
    images_with_frame_names = []
    for i in range(1, len(path_to_images)):
        print('iter: ', i, '/', len(path_to_images))
        first_image = cv2.imread(path_to_images[i - 1], 0)
        first_image = first_image / 255
        second_image = cv2.imread(path_to_images[i], 0)
        second_image = second_image / 255
        points_with_velocity = calc_interesting_point_with_velocity(first_image, second_image)
        first_image_gray = create_color_gray(first_image)
        visual_threshold_moving_points(first_image_gray, points_with_velocity, thr=0.04)
        # images_with_frame_names.append((first_image_gray, os.path.basename(path_to_images[i - 1])))
        images_with_frame_names.append((first_image_gray, 'frame_' + str(i - 1) + '.jpg'))
    for (frame, name) in images_with_frame_names:
        # cv2.imshow(name, frame)
        cv2.imwrite(os.path.join(PATH_TO_SAVE, name), normalize_image(frame, 0, 255).astype(np.uint8))
        cv2.waitKey()


if __name__ == '__main__':
    main()
