# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowuBGHQb.ui'
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
    QLayout, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QVBoxLayout, QWidget)

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
        self.verify_pushbutton = QPushButton(self.widget)
        self.verify_pushbutton.setObjectName(u"verify_pushbutton")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.verify_pushbutton.sizePolicy().hasHeightForWidth())
        self.verify_pushbutton.setSizePolicy(sizePolicy)
        self.verify_pushbutton.setMinimumSize(QSize(150, 50))
        font = QFont()
        font.setPointSize(12)
        self.verify_pushbutton.setFont(font)
        self.verify_pushbutton.setLayoutDirection(Qt.LeftToRight)
        self.verify_pushbutton.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        self.verticalLayout_4.addWidget(self.verify_pushbutton, 0, Qt.AlignHCenter)

        self.register_pushbutton = QPushButton(self.widget)
        self.register_pushbutton.setObjectName(u"register_pushbutton")
        sizePolicy.setHeightForWidth(self.register_pushbutton.sizePolicy().hasHeightForWidth())
        self.register_pushbutton.setSizePolicy(sizePolicy)
        self.register_pushbutton.setMinimumSize(QSize(150, 50))
        self.register_pushbutton.setFont(font)
        self.register_pushbutton.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        self.verticalLayout_4.addWidget(self.register_pushbutton, 0, Qt.AlignHCenter)

        self.registered_pushbutton = QPushButton(self.widget)
        self.registered_pushbutton.setObjectName(u"registered_pushbutton")
        sizePolicy.setHeightForWidth(self.registered_pushbutton.sizePolicy().hasHeightForWidth())
        self.registered_pushbutton.setSizePolicy(sizePolicy)
        self.registered_pushbutton.setMinimumSize(QSize(150, 50))
        self.registered_pushbutton.setFont(font)
        self.registered_pushbutton.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        self.verticalLayout_4.addWidget(self.registered_pushbutton, 0, Qt.AlignHCenter)


        self.horizontalLayout_4.addWidget(self.widget)

        self.camera_frame = QFrame(self.centralwidget)
        self.camera_frame.setObjectName(u"camera_frame")
        self.camera_frame.setBaseSize(QSize(0, 0))
        self.camera_frame.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.camera_frame.setFrameShape(QFrame.StyledPanel)
        self.verticalLayout = QVBoxLayout(self.camera_frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(9, 0, 9, 0)
        self.camera_label = QLabel(self.camera_frame)
        self.camera_label.setObjectName(u"camera_label")
        self.camera_label.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.camera_label.sizePolicy().hasHeightForWidth())
        self.camera_label.setSizePolicy(sizePolicy1)
        self.camera_label.setMinimumSize(QSize(640, 360))
        self.camera_label.setFrameShape(QFrame.NoFrame)
        self.camera_label.setText(u"")
        self.camera_label.setTextFormat(Qt.PlainText)
        self.camera_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.camera_label)

        self.register_frame = QWidget(self.camera_frame)
        self.register_frame.setObjectName(u"register_frame")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.register_frame.sizePolicy().hasHeightForWidth())
        self.register_frame.setSizePolicy(sizePolicy2)
        self.register_frame.setMinimumSize(QSize(0, 16))
        self.register_frame.setFont(font)
        self.register_frame.setAutoFillBackground(False)
        self.horizontalLayout_2 = QHBoxLayout(self.register_frame)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setContentsMargins(-1, 0, 9, 7)
        self.take_picture_pushbutton = QPushButton(self.register_frame)
        self.take_picture_pushbutton.setObjectName(u"take_picture_pushbutton")
        sizePolicy1.setHeightForWidth(self.take_picture_pushbutton.sizePolicy().hasHeightForWidth())
        self.take_picture_pushbutton.setSizePolicy(sizePolicy1)
        self.take_picture_pushbutton.setMinimumSize(QSize(0, 0))
        self.take_picture_pushbutton.setFont(font)

        self.horizontalLayout_2.addWidget(self.take_picture_pushbutton)

        self.save_picture_pushbutton = QPushButton(self.register_frame)
        self.save_picture_pushbutton.setObjectName(u"save_picture_pushbutton")
        self.save_picture_pushbutton.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.save_picture_pushbutton.sizePolicy().hasHeightForWidth())
        self.save_picture_pushbutton.setSizePolicy(sizePolicy1)
        self.save_picture_pushbutton.setFont(font)

        self.horizontalLayout_2.addWidget(self.save_picture_pushbutton)


        self.verticalLayout.addWidget(self.register_frame, 0, Qt.AlignVCenter)


        self.horizontalLayout_4.addWidget(self.camera_frame, 0, Qt.AlignHCenter)

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
        self.verify_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Verify", None))
        self.register_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Register", None))
        self.registered_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Registered folder", None))
        self.take_picture_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Take Picture", None))
        self.save_picture_pushbutton.setText(QCoreApplication.translate("MainWindow", u"Save picture", None))
    # retranslateUi

