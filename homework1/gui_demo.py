# -*- coding=utf-8 -*-

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import sys



class AStarUI(QMainWindow):

    def __init__(self):
        super(AStarUI, self).__init__()
        self.setGeometry(0, 0, 1100, 1100)
        self.setWindowTitle("A* algorithm")
        self.coups = QTableWidget(1,3,self)
        self.coups.setItem(0, 0, QTableWidgetItem(0))
        self.coups.setItem(0, 1, QTableWidgetItem(0))
        self.coups.setItem(0, 2, QTableWidgetItem(0))
        self.coups.setHorizontalHeaderItem(0, QTableWidgetItem("asd"))
        self.coups.setHorizontalHeaderItem(1, QTableWidgetItem("aerg"))
        self.coups.setHorizontalHeaderItem(2, QTableWidgetItem("erg"))
        self.coups.move(500,500)
        self.coups.setFixedSize(600, 240)
        self.qp = QtGui.QPainter()
        self.qp.drawPixmap(200, 200, 8, 8)
        self.show()



app = QApplication(sys.argv)
window = AStarUI()
app.exec()

# app = QtWidgets.QApplication([]) # [] 参数
# label = QtWidgets.QLabel("hello world")
# label.show()
# app.exec()


