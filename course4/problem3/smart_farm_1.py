import datetime
import random
import threading
import time


class ParmSensor:
    def __init__(self, sensor_name):
        self.sensor_name = sensor_name
        self.temperature = 0
        self.illuminance = 0
        self.humidity = 0

    def set_data(self):
        self.temperature = random.randint(20, 30)
        self.illuminance = random.randint(5000, 10000)
        self.humidity = random.randint(40, 70)

    def get_data(self):
        return self.temperature, self.illuminance, self.humidity


def run_sensor(sensor):
    while True:
        sensor.set_data()

        temperature, illuminance, humidity = sensor.get_data()
        now = datetime.datetime.now()

        print(
            f"{now.strftime('%Y-%m-%d %H:%M:%S')} "
            f'{sensor.sensor_name} - '
            f'temp {temperature}, '
            f'light {illuminance}, '
            f'humi {humidity}'
        )

        time.sleep(10)


def main():
    sensors = []

    for number in range(1, 6):
        sensor = ParmSensor(f'Parm-{number}')
        sensors.append(sensor)

    for sensor in sensors:
        thread = threading.Thread(target=run_sensor, args=(sensor,))
        thread.daemon = True
        thread.start()

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()

    