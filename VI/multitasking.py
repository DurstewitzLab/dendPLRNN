import multiprocessing as mp
import subprocess


class Task:
    def __init__(self, command, name):
        self.command = command
        self.name = name


class Argument:
    def __init__(self, name, values, add_to_name_as=None):
        self.name = name
        self.values = values
        self.add_to_name_as = add_to_name_as
        if len(values) > 1:
            print_statement = 'please specify a name addition for argument {}, because it has more than one value'.format(
                name)
            if name != 'run':
                assert add_to_name_as is not None, print_statement


def add_argument(tasks, arg):
    new_tasks = []
    for task in tasks:
        for arg_value in arg.values:
            if type(arg_value) is list:
                arg_name = '-'.join([str(i) for i in arg_value])
                new_name = add_to_name(task, arg, arg_name)
                arg_command = ' '.join([str(i) for i in arg_value])
                new_command = " ".join([task.command, "--{}".format(arg.name), arg_command])
            else:
                new_name = add_to_name(task, arg, arg_value)
                new_command = " ".join([task.command, "--{}".format(arg.name), str(arg_value)])
            new_tasks.append(Task(new_command, new_name))
    return new_tasks


def add_to_name(task, arg, arg_value):
    if arg.add_to_name_as is not None:
        new_name = "".join([task.name, arg.add_to_name_as, str(arg_value).zfill(2)])
    else:
        new_name = task.name
    return new_name


def create_tasks_from_arguments(args):
    tasks = [Task(command='python3 main.py', name='')]
    for arg in args:
        tasks = add_argument(tasks, arg)

    for task in tasks:
        task.command = " ".join([task.command, '--name', task.name])
    return tasks


def run_settings(tasks, n_cpu):
    pool = mp.Pool(processes=n_cpu)
    pool.map(process_task, tasks)
    pool.close()
    pool.join()


def process_task(task):
    subprocess.call(task.command.split())