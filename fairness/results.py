import pathlib

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
    
