import argparse
from get_config import get_config
from task import Classify_task
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)


args = parser.parse_args()

config = get_config(args.config_file)


task=Classify_task(config)
task.training() #traning, khi nào muốn predict thì cmt lại
task.evaluate() #đánh giá trên test data