import os

import numpy as np
import cv2

from video_util import convert_frames_to_video

PATH_TO_DUMPTRUCK = 'examples/video_data'
PATH_TO_SAVE = 'examples/crooped_video_data'


def normalize_image(x: np.ndarray, a: float, b: float):
    min_val = np.min(x)
    max_val = np.max(x)
    return a + (x - min_val) * (b - a) / (max_val - min_val)


def convolve(image: np.ndarray, kernel: np.ndarray):
    h_kernel, w_kernel = kernel.shape
    h_image, w_image = image.shape
    h_offset = h_kernel // 2
    w_offset = w_kernel // 2
    padded_image = np.zeros((h_image + h_offset * 2, w_image + w_offset * 2))
    padd_h_image = padded_image.shape[0] - h_offset
    padd_w_image = padded_image.shape[1] - w_offset
    padded_image[h_offset:padd_h_image, w_offset:padd_w_image] = image
    output = np.zeros_like(padded_image)
    for i in range(h_offset, h_image):
        for j in range(w_offset, w_image):
            output[i, j] = np.sum(kernel * padded_image[i - h_offset: i + h_offset + 1, j - w_offset: j + w_offset + 1])
    return output[h_offset:padd_h_image, w_offset:padd_w_image]


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


def create_color_gray(picture: np.ndarray):
    color_gray = np.zeros((picture.shape[0], picture.shape[1], 3))
    for i in range(color_gray.shape[0]):
        for j in range(color_gray.shape[1]):
            color_gray[i][j] = [picture[i][j], picture[i][j], picture[i][j]]
    return color_gray


def draw_arrow(image: np.ndarray, p: (int, int), q: (int, int), color: (np.uint8, np.uint8, np.uint8),
               arrow_magnitude: int = 9, thickness: int = 1, line_type: int = 8, shift: int = 0) -> None:
    cv2.line(image, p, q, color, thickness, line_type, shift)
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)


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
        vel_x, vel_y = np.dot(np.dot(A_pinv, A_trans), It)  # we have the vectors with minimized square error
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
        #cv2.imshow(name, frame)
        cv2.imwrite(os.path.join(PATH_TO_SAVE, name), normalize_image(frame, 0, 255).astype(np.uint8))
        cv2.waitKey()


if __name__ == '__main__':
    main()
