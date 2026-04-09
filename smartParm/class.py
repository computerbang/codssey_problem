import random
import time
import datetime
import threading

print_lock = threading.Lock()


class ParmSensor:
    def __init__(self, name):
        self.name = name
        self.setData()

    def setData(self):
        self.temp = random.randint(20, 30)
        self.light = random.randint(5000, 10000)
        self.humid = random.randint(40, 70)

    def getData(self):
        return self.temp, self.light, self.humid


def sensor_work(sensor):
    while True:
        sensor.setData()
        temp, light, humid = sensor.getData()

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with print_lock:
            print(f"{now} {sensor.name} - temp {temp:02d}, light {light:05d}, humi {humid:02d}")

        time.sleep(10)


def main():
    sensors = [ParmSensor(f"Parm-{i}") for i in range(1, 6)]
    threads = []

    for sensor in sensors:
        t = threading.Thread(target=sensor_work, args=(sensor,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()