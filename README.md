# ML_template  
Template repository to build machine learning project with Pytorch.
This repository does not support Trainer class.  
You can copy this repository and modify the following files.  
- `src/train.py`
- `src/dataset.py`
- `src/prepro.py`
- `src/loss.py`
- `src/net.py`


## Directory Structure
```
ML_template
├── LICENSE
├── README.md
├── data                  <- Directory to store train/val data.
├── log                   <- Directory to store log files.
├── model                 <- Directory to store trained models.
├── path.sh                
├── run.sh                 
├── src                    
│   └── __init__.py
└── tools
    ├── Makefile
    ├── commands          <- Util shellscripts like parse_options.sh
    └── requirements.txt  
```

## Supported utility functionalities.
### CustomLogClass
You can easily use this logger class with the following code.

```
from writer import Logger

log_name = 'TestLogName'
logger = Logger(log_name, 'enc', 'tbx_log', 'slack')

# you can use info() as normal logger function.
logger.enc.info('Encoder Log!')

# figure() supports log from dictionary
phase='train
dic = {
    'loss_1': 0.1,
    'loss_2': 1.2
}
logger.tbx_log.figure(
    phase,
    dic,
    iter_count=100
)

# each functions can send log text to slack channel.
logger.slack.info(
    'This message will send to slack channel',
    send_to_slack=True
)
```
Log files will save into `log/log/${log_name}`.  
Figures like losses will be written in tensorboard event file in `log/tbx/${log_name}`  
Every figures will be named as `${phase}/{key}`
  
If you want to use slack functionality, you have to set
api token and channel name in `path.sh`

Log format is `[%(asctime)s] - %(name)s - %(message)s - ${logger_name} - ${text}`  
Sample output:
```
[%(asctime)s] - %(name)s - %(message)s - enc - Hello
[%(asctime)s] - %(name)s - %(message)s - slack - This message will send to sklack channel.
```

## Coinfig class
will support gin config.
