import argparse
import time
import cv2
from tracker import Krolik
from utils import gen_params


def get_args():
    parser = argparse.ArgumentParser(description='Little pyatachok object Tracker.')
    parser.add_argument('--path', default="data/input.mp4", help="Путь до видеофайла")
    parser.add_argument('--params', default="config/params.yaml", help="Путь до yaml файла с параметрами")
    parser.add_argument('--debug', action='store_true', help="Дебаг для вывода таймингов")
    return parser.parse_args()


def show_draw(img, state, color=(0, 255, 0)):
    """
    Функция отрисовки информации и положения объекта наблюдения
    :param img:
    :param state:
    :param color:
    :return:
    """
    if state is not None:
        state = [int(s) for s in state['target_bbox']]
        cv2.rectangle(
            img, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), color, 5)
    font_color = (0, 0, 0)
    cv2.putText(
        img, 'Tracking', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
    cv2.putText(
        img, 'Press S to reinitialize', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
    cv2.putText(
        img, 'Press Q or ESC to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
    return img


def inference(args, tracker):
    """

    :param video_path:
    :param tracker:
    :return:
    """
    cap = cv2.VideoCapture(args.path)
    display_name = "Demo"

    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)

    tracking_started = False
    state_dict = None
    while True:
        print(f"\nFrame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
        _, frame = cap.read()

        if frame is None:
            break

        frame_temp = frame.copy()

        if tracking_started:
            t1 = time.time()
            # Трекинг объекта
            state_dict = tracker.track(frame)
            t2 = time.time()
            print(f"TRACK| TIME: {t2 - t1} FPS: {1 / (t2 - t1)}")

        # Отрисовка для вывода на экран
        frame_show = show_draw(frame_temp.copy(), state_dict)
        cv2.imshow(display_name, frame_show)
        key = cv2.waitKey(30)

        if key == ord('q') or key == 27:
            # Завершение программы
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            # Инициализация объекта наблюдения
            cv2.putText(
                frame_temp, 'Select target ROI and press ENTER',
                (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                (0, 0, 0), 1)

            # Получение координат
            x, y, w, h = cv2.selectROI(display_name, frame_temp, fromCenter=False)
            state_dict = tracker.initialize(frame, {'init_bbox': [x, y, w, h]})
            tracking_started = True


if __name__ == '__main__':
    args = get_args()

    params = gen_params(args.params)
    tracker = Krolik(params, args.debug)

    inference(args, tracker)
