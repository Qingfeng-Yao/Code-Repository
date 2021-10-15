# _*_ coding: utf-8 _*_
import logging
import os
import time


class Logger(object):
    def __init__(self, logger, dirpath):

        # ����һ��logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # ������־����
        filename = time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # ����һ��handler������д����־
        fh = logging.FileHandler(dirpath+'/'+filename+'.log')
        fh.setLevel(logging.INFO)

        # ����һ��handler���������������̨
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # ����handler�������ʽ
        formatter = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)s] - %(levelname)s - %(message)s ')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # ��logger���handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger