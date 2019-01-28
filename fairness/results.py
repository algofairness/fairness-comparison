import pathlib
import os
import tempfile
import shutil

from fairness.metrics.list import get_metrics

# FIXME: this could probably be handled better on Windows
def local_results_path():
    home = pathlib.Path.home()
    path = home / '.fairness'
    ensure_dir(path)
    return path

def ensure_dir(path):
    if path.exists() and not path.is_dir():
        raise Exception("Cannot run fairness: local storage location %s is not a directory" % path)
    path.mkdir(parents=True, exist_ok=True)

##############################################################################

def get_metrics_list(dataset, sensitive_dict, tag):
    return [metric.get_name() for metric in get_metrics(dataset, sensitive_dict, tag)]

def get_detailed_metrics_header(dataset, sensitive_dict, tag):
    return ','.join(['algorithm', 'params', 'run-id'] + get_metrics_list(dataset, sensitive_dict, tag))
    
class ResultsFile(object):

    def __init__(self, filename, dataset, sensitive_dict, tag):
        self.filename = filename
        self.dataset = dataset
        self.sensitive_dict = sensitive_dict
        self.tag = tag
        handle, name = self.create_new_file()
        self.fresh_file = handle
        self.temp_name = name

    def create_new_file(self):
        fd, name = tempfile.mkstemp()
        os.close(fd)
        f = open(name, "w")
        f.write(get_detailed_metrics_header(
            self.dataset, self.sensitive_dict, self.tag) + '\n')
        return f, name

    def write(self, *args):
        self.fresh_file.write(*args)
        self.fresh_file.flush()
        os.fsync(self.fresh_file.fileno())

    def close(self):
        # FIXME, this is where we merge the new results to the old ones
        # for now, we just atomically move the new file onto the old one.
        self.fresh_file.close()
        shutil.move(self.temp_name, self.filename)
        
