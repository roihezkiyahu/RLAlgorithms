import time
import os
import pandas as pd


class Logger:
    def __init__(self, apply=True, verbose=True, output_folder=""):
        self.timers = dict()
        self.logs = []
        self.id_counter = 1
        self.verbose = verbose
        self.apply = apply
        self.output_folder = output_folder

    def start_timer(self, name):
        self.timers[name] = time.time()

    def stop_timer(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer {name} was not started")

        start_time = self.timers[name]
        end_time = time.time()
        execution_time = end_time - start_time

        if self.verbose:
            print(f"Timer {name} executed in {execution_time} seconds")

        self.logs.append({
            'id': self.id_counter,
            'name': name,
            'execution_time': execution_time
        })

        del self.timers[name]
        self.id_counter += 1

    def get_logs(self):
        return self.logs

    def time_and_log(self, func, name, *args, **kwargs):
        if not self.apply:
            to_return = func(*args, **kwargs)
            return to_return
        self.start_timer(name)
        to_return = func(*args, **kwargs)
        self.stop_timer(name)
        return to_return


    def dump_log(self, file_name):
        if (self.output_folder == "") or (not self.apply):
            print("no logger output path provided")
            return
        if not file_name.endswith(".csv"):
            file_name = file_name + ".csv"
        full_path = os.path.join(self.output_folder, file_name)
        self.validate_output_path(os.path.dirname(full_path))
        pd.DataFrame(self.logs).to_csv(full_path)

    @staticmethod
    def validate_output_path(path):
        if not os.path.exists(path):
            os.makedirs(path)
