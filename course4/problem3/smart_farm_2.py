import datetime
import random
import subprocess
import threading
import time


MYSQL_USER = 'root'
MYSQL_PASSWORD = '1234'
MYSQL_DATABASE = 'smart_farm'


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


def run_mysql_query(query):
    command = [
        'mysql',
        '-u',
        MYSQL_USER,
        f'-p{MYSQL_PASSWORD}',
        MYSQL_DATABASE,
        '-e',
        query,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print('MySQL error:', result.stderr.strip())


def insert_sensor_data(input_time, temperature, illuminance, humidity):
    formatted_time = input_time.strftime('%Y-%m-%d %H:%M:%S')

    query = (
        'INSERT INTO parm_data '
        '(input_time, temperature, illuminance, humidity) '
        'VALUES '
        f"('{formatted_time}', {temperature}, {illuminance}, {humidity});"
    )

    run_mysql_query(query)


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

        insert_sensor_data(
            now,
            temperature,
            illuminance,
            humidity,
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