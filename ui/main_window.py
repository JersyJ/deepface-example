# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowKhgmaS.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(880, 573)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_4 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_4 = QVBoxLayout(self.widget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verify_pushButton = QPushButton(self.widget)
        self.verify_pushButton.setObjectName(u"verify_pushButton")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.verify_pushButton.sizePolicy().hasHeightForWidth())
        self.verify_pushButton.setSizePolicy(sizePolicy)
        self.verify_pushButton.setMinimumSize(QSize(150, 50))
        self.verify_pushButton.setLayoutDirection(Qt.LeftToRight)
        self.verify_pushButton.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        self.verticalLayout_4.addWidget(self.verify_pushButton, 0, Qt.AlignHCenter)

        self.register_pushButton = QPushButton(self.widget)
        self.register_pushButton.setObjectName(u"register_pushButton")
        sizePolicy.setHeightForWidth(self.register_pushButton.sizePolicy().hasHeightForWidth())
        self.register_pushButton.setSizePolicy(sizePolicy)
        self.register_pushButton.setMinimumSize(QSize(150, 50))

        self.verticalLayout_4.addWidget(self.register_pushButton, 0, Qt.AlignHCenter)

        self.registered_pushButton = QPushButton(self.widget)
        self.registered_pushButton.setObjectName(u"registered_pushButton")
        sizePolicy.setHeightForWidth(self.registered_pushButton.sizePolicy().hasHeightForWidth())
        self.registered_pushButton.setSizePolicy(sizePolicy)
        self.registered_pushButton.setMinimumSize(QSize(150, 50))

        self.verticalLayout_4.addWidget(self.registered_pushButton, 0, Qt.AlignHCenter)


        self.horizontalLayout_4.addWidget(self.widget)

        self.camera_frame = QFrame(self.centralwidget)
        self.camera_frame.setObjectName(u"camera_frame")
        self.camera_frame.setBaseSize(QSize(0, 0))
        self.camera_frame.setFrameShape(QFrame.NoFrame)
        self.horizontalLayout = QHBoxLayout(self.camera_frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.camera_label = QLabel(self.camera_frame)
        self.camera_label.setObjectName(u"camera_label")
        self.camera_label.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.camera_label.sizePolicy().hasHeightForWidth())
        self.camera_label.setSizePolicy(sizePolicy1)
        self.camera_label.setMinimumSize(QSize(640, 360))
        self.camera_label.setText(u"")
        self.camera_label.setTextFormat(Qt.PlainText)

        self.horizontalLayout.addWidget(self.camera_label)


        self.horizontalLayout_4.addWidget(self.camera_frame)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 880, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.verify_pushButton.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.register_pushButton.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.registered_pushButton.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
    # retranslateUi

