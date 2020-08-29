# custom log utils class
import os
from slacker import Slacker

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

class CustomLogClass(object):
    def __init__(self, name, logger, writer):
        self.name = name
        self.logger = logger
        self.writer = writer
        self.base_config = base_config
        self.sender = SlackLogSender()

    def info(self, text, send_to_slack=False):
        if send_to_slack:
            self.sender.send(text)
        self.logger.info(text)

    def figure(self, phase, dic, iter_count, send_to_slack=False):
        # function to log figures like loss.
        for k,v in dic.items():
            self.writer.add_scalar('data/%s/%s' % (phase, k), v, iter_count)
        self.writer.flush()
        self.sender.send_dic(phase, dic, iter_count)


class SlackLogSender(object):
    def __init__(self):
        self.slack = Slacker(os.environ['SLACK_API_TOKEN'])
        self.channel = os.environ['SLACK_CHANNEL']

    def send(self, text):
        self.slack.chat.post_message(
            self.channel,
            text
        )

    def send_dic(self, phase, dic, iter_count):
        t = '[ITERATION %d]\n' % iter_count
        t += '```\n'
        for k,v in dic.items():
            t += '%s/%s : %s \n' % (phase, k, v)
        t += '```'
        self.sender.chat.post_message(t)
