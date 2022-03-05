""" Entry point for the graphical user interface """

##### local run, set module load path ###### 
#1, export PYTHONPATH=/root/fmi/FMPyOrth/:/usr/local/lib/python3.6/dist-packages
#2, sundials library copy to local fmpy folder
#3, python3 -m fmpy.gui


if __name__ == '__main__':


    import os
    import sys
    import ctypes
    import platform
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QApplication
    from fmpy.gui.MainWindow import MainWindow

    if os.name == 'nt' and int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)

    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    for i, v in enumerate(sys.argv[1:]):
        if i > 0:
            window = MainWindow()
            window.show()
        window.load(v)

    sys.exit(app.exec_())
