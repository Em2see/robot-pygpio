import time
import pigpio
from typing import Any
import cv2
import tensorflow as tf
from enum import Enum
import numpy as np


MOTOR_LEFT_PIN = 18
MOTOR_RIGHT_PIN = 26
SENSOR_DISTANCE_PIN = 28
LED_RED_PIN = 0
LED_GREEN_PIN = 0
CAMERA_DEV = 0

MOTOR_MAX_SPEED = 100


class Direction(str, Enum):
    FORWAD = 'forward'
    BACKWARD = 'backward'


class LedState(str, Enum):
    ON = "on"
    OFF = "off"


class Robot:
    def __init__(self):
        self.pi = pigpio.pi()
        self.pi.set_mode(MOTOR_LEFT_PIN, pigpio.OUTPUT)
        self.pi.set_mode(MOTOR_RIGHT_PIN, pigpio.OUTPUT)

        self.pi.set_mode(SENSOR_DISTANCE_PIN, pigpio.INPUT)
    
    def __exit__(self):
        self.pi.stop()

    def _set_motor_speed(self, motor_pin: int, speed: int):
        speed = min(speed, MOTOR_MAX_SPEED)
        self.pi.set_servo_pulsewidth(motor_pin, speed)

    def _set_motor_direction(self, motor_pin: int, direction: Direction):
        if direction == Direction.FORWAD:
            self.pi.set_servo_pulsewidth(motor_pin, 1500)
        elif direction == Direction.BACKWARD:
            self.pi.set_servo_pulsewidth(motor_pin, 500)
        else:
            self.pi.set_servo_pulsewidth(motor_pin, 0)
 
    @property
    def get_distance(self, pin: int) -> int:
        # Функции для управления датчиками
        return self.pi.read(pin)

    def set_led_state(self, pin: int, led_state: LedState):
        # Функции для управления светодиодами
        if led_state == LedState.ON:
            self.pi.write(pin, 1)
        elif led_state == LedState.OFF:
            self.pi.write(pin, 0)

    def motor_temperature(self, pin: int) -> int:
        # Функции для получения информации о состоянии электроники
        return self.pi.get_temperature(pin)

    def led_state(self, pin:int):
        return self.pi.read(pin)

    @property
    def state(self) -> dict[str, Any]:
        """
        Возвращает словарь со следующей информацией о состоянии робота:

        * motor_temperatures: температура моторов
        * led_states: состояние светодиодов
        * distance: расстояние до препятствия
        """
        motor_temperatures = {
            "left_motor": self.motor_temperature(MOTOR_LEFT_PIN),
            "right_motor": self.motor_temperature(MOTOR_RIGHT_PIN),
        }
        led_states = {
            "red_led": self.led_state(LED_RED_PIN),
            "green_led": self.led_state(LED_GREEN_PIN),
        }
        distance = self.get_distance(SENSOR_DISTANCE_PIN)

        return {
            "motor_temperatures": motor_temperatures,
            "led_states": led_states,
            "distance": distance,
        }

    def get_lidar_image(self):
        # Получаем изображение с лидар камеры
        image = self.lidar_camera.read()
        return image

    def process_image_from_lidar(self, image):
        # Обрабатываем изображение с лидар камеры

        # Получаем точки препятствий

        obstacles = self.lidar_camera.get_obstacles(image)

        # Получаем центр ближайшего препятствия

        center = obstacles[0]

        # Рисуем центр препятствия на изображении

        image = cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        return image

    def get_state_from_lidar_image(self, image) -> tuple:
        # Получаем состояние робота на основе изображения с лидар камеры

        # Получаем центр ближайшего препятствия

        center = self.process_image_from_lidar(image)

        # Возвращаем состояние робота

        return center[0], center[1]

    def get_image(self):
        """
        Возвращает изображение с камеры.
        """
        camera = cv2.VideoCapture(CAMERA_DEV)
        _, frame = camera.read()
        camera.release()
        return frame

    @classmethod
    def process_image(cls, image):
        """
        Обрабатывает изображение с камеры.
        """

        # Переводим изображение в формат RGB

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Определяем объекты на изображении

        objects = tf.image.segmentation.semantic_segmentation(image, ["background", "obstacle"])

        # Получаем центр объекта

        center = tf.reduce_sum(objects[..., 0] * tf.cast(objects[..., 1] != 0, dtype=tf.float32), axis=(1, 2)) / tf.reduce_sum(objects[..., 1] != 0, axis=(1, 2))

        # Рисуем центр объекта на изображении

        image = cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        return image
    

    def perform_action(self, state, agent):
        """
        Выполняет действие робота.

        Args:
            state: Текущее состояние робота.

        Returns:
            Награда, следующее состояние робота и флаг, указывающий на то, достиг ли робот цели.
        """

        # Получаем действие от агента
        action = agent.act(state)

        # Выполняем действие
        if action == 0:
            # Движение вперед
            self._set_motor_speed(MOTOR_LEFT_PIN, 100)
            self._set_motor_speed(MOTOR_RIGHT_PIN, 100)
        elif action == 1:
            # Движение назад
            self._set_motor_speed(MOTOR_LEFT_PIN, -100)
            self._set_motor_speed(MOTOR_RIGHT_PIN, -100)
        elif action == 2:
            # Вращение влево
            self._set_motor_speed(MOTOR_LEFT_PIN, -100)
            self._set_motor_speed(MOTOR_RIGHT_PIN, 100)
        else:
            # Вращение вправо
            self._set_motor_speed(MOTOR_LEFT_PIN, 100)
            self._set_motor_speed(MOTOR_RIGHT_PIN, -100)

        # Получаем следующее состояние робота
        next_state = self.get_next_state(state, action)

        # Получаем награду
        reward = self.get_reward(state, action, next_state)

        # Проверяем, достиг ли робот цели
        done = self.is_done(state, next_state)

        return reward, next_state, done


    def navigate_and_map(self, agent, start_state):
        """
        Перемещает робота по квартире и составляет карту.

        Args:
            pi: Объект GPIO.
            agent: Агент Q-обучения.
            start_state: Начальное состояние.

        Returns:
            Карта квартиры.
        """

        # Инициализация переменных

        map = np.zeros((480, 640))
        state = start_state
        done = False
        steps = 0

        # Цикл перемещения робота

        while not done:
            # Получаем изображение с камеры
            image = self.get_image()

            # Обрабатываем изображение
            image = self.process_image(image)

            # Получаем текущее состояние робота
            state = agent.get_state(image)

            # Получаем действие от агента
            action = agent.act(state)

            # Выполняем действие
            reward, next_state = self.perform_action(state, agent)

            # Обновляем карту
            map[state[0], state[1]] = reward

            # Задержка
            time.sleep(0.1)

        # Сохраняем карту
        np.save("map.npy", map)

        return map
