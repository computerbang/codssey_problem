import datetime
import queue
import random
import subprocess
import threading
import time

import matplotlib.pyplot as plt


MYSQL_USER = 'root'
MYSQL_PASSWORD = '1234'
MYSQL_DATABASE = 'smart_farm'

SENSOR_COUNT = 5
SENSOR_INTERVAL = 10
DB_CHECK_INTERVAL = 1

sensorQ = queue.Queue()
stop_event = threading.Event()


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

    return result.stdout


def insert_sensor_data(
        sensor_name,
        input_time,
        temperature,
        illuminance,
        humidity):
    formatted_time = input_time.strftime('%Y-%m-%d %H:%M:%S')

    query = (
        'INSERT INTO parm_data '
        '(sensor_name, input_time, temperature, illuminance, humidity) '
        'VALUES '
        f"('{sensor_name}', '{formatted_time}', "
        f'{temperature}, {illuminance}, {humidity});'
    )

    run_mysql_query(query)


def get_sensor_data():
    query = (
        'SELECT '
        'sensor_name, '
        'input_time, '
        'temperature, '
        'illuminance, '
        'humidity '
        'FROM parm_data '
        'ORDER BY input_time ASC;'
    )

    output = run_mysql_query(query)
    lines = output.strip().splitlines()

    if len(lines) <= 1:
        return []

    sensor_data = []

    for line in lines[1:]:
        columns = line.split('\t')

        if len(columns) != 5:
            continue

        sensor_data.append({
            'sensor_name': columns[0],
            'input_time': columns[1],
            'temperature': int(columns[2]),
            'illuminance': int(columns[3]),
            'humidity': int(columns[4]),
        })

    return sensor_data


def run_sensor(sensor):
    while not stop_event.is_set():
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

        sensor_data = {
            'sensor_name': sensor.sensor_name,
            'input_time': now,
            'temperature': temperature,
            'illuminance': illuminance,
            'humidity': humidity,
        }

        sensorQ.put(sensor_data)

        time.sleep(SENSOR_INTERVAL)


def run_database_worker():
    while not stop_event.is_set():
        if not sensorQ.empty():
            sensor_data = sensorQ.get()

            insert_sensor_data(
                sensor_data['sensor_name'],
                sensor_data['input_time'],
                sensor_data['temperature'],
                sensor_data['illuminance'],
                sensor_data['humidity'],
            )

            print(
                'Saved to DB:',
                sensor_data['sensor_name'],
                sensor_data['input_time'].strftime('%Y-%m-%d %H:%M:%S'),
            )

            sensorQ.task_done()

        time.sleep(DB_CHECK_INTERVAL)


def get_hourly_temperature_average():
    data = get_sensor_data()
    result = {}

    for row in data:
        if row['sensor_name'] == 'Unknown':
            continue

        hour_key = row['input_time'][:13] + ':00:00'
        key = (row['sensor_name'], hour_key)

        if key not in result:
            result[key] = {
                'sensor_name': row['sensor_name'],
                'hour_time': hour_key,
                'temperature_sum': 0,
                'count': 0,
                'max_humidity': 0,
            }

        result[key]['temperature_sum'] += row['temperature']
        result[key]['count'] += 1

        if row['humidity'] > result[key]['max_humidity']:
            result[key]['max_humidity'] = row['humidity']

    averages = []

    for value in result.values():
        avg_temperature = value['temperature_sum'] / value['count']

        averages.append({
            'sensor_name': value['sensor_name'],
            'hour_time': value['hour_time'],
            'avg_temperature': avg_temperature,
            'max_humidity': value['max_humidity'],
        })

    return averages


def draw_temperature_graph():
    data = get_hourly_temperature_average()

    if not data:
        print('No data for graph.')
        return

    labels = []
    temperatures = []
    humidity_points_x = []
    humidity_points_y = []

    for row in data:
        label = f"{row['sensor_name']}\n{row['hour_time']}"
        labels.append(label)
        temperatures.append(row['avg_temperature'])

    for index, row in enumerate(data):
        if row['max_humidity'] > 90:
            humidity_points_x.append(index)
            humidity_points_y.append(row['avg_temperature'])

    plt.figure(figsize=(12, 6))
    plt.plot(labels, temperatures, marker='o', label='Average Temperature')

    if humidity_points_x:
        plt.scatter(
            humidity_points_x,
            humidity_points_y,
            color='red',
            marker='*',
            s=150,
            label='Humidity over 90%',
        )

    plt.title('Sensor Hourly Average Temperature')
    plt.xlabel('Sensor / Hour')
    plt.ylabel('Average Temperature')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    sensors = []

    for number in range(1, SENSOR_COUNT + 1):
        sensor = ParmSensor(f'Parm-{number}')
        sensors.append(sensor)

    threads = []

    for sensor in sensors:
        thread = threading.Thread(target=run_sensor, args=(sensor,))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    db_thread = threading.Thread(target=run_database_worker)
    db_thread.daemon = True
    db_thread.start()
    threads.append(db_thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

        for thread in threads:
            thread.join(timeout=1)

        print()
        print('Program stopped.')
        print('Drawing graph...')
        draw_temperature_graph()


if __name__ == '__main__':
    main()