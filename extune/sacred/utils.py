import logging, os

def config_logger(run_dir: str) -> None:
    '''
    Configuring the logger to be exactly as one wishes can be verbose.
    Eschewed away here for clarity.

    Arguments:
        run_dir (str): the directory to which the the log file 'experiment.log'
            should be written
    Returns:
        None
    '''
    print("sacred run_dir", run_dir)
    logging.basicConfig(
        filename = os.path.join(run_dir, 'experiment.log'),
        level    = logging.DEBUG,
        format   = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
