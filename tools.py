# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   BeginningPython
@File       :   tools.py
@Version    :   v0.1
@Time       :   2020-12-25 18:00
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""


def show_subtitle(message):
    # 输出运行模块的子标题
    print('-' * 15, '>' + message + '<', '-' * 15)
    pass


def show_title(message):
    # 输出运行模块的子标题
    print()
    print('=' * 15, '>' + message + '<', '=' * 15)
    pass


def beep_end():
    # 运行结束的提醒
    # 没有喇叭就没法正常导入
    frequency = 600
    duration = 500
    try:
        import winsound
    except ImportError:
        # Linux 无法正确进行语音提醒
        # print("\a")
        import os
        # apt-get install beep
        # os.system("beep -f %s -l %s" % (frequency, duration))
    else:
        winsound.Beep(frequency, duration)
    pass


def show_figures():
    import matplotlib.pyplot as plt
    # 运行结束前显示存在的图形
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass


def get_root_path():
    # 提供项目路径
    import os
    return os.path.dirname(__file__) + "\\"


if __name__ == '__main__':
    show_title("program begin")
    get_root_path()
