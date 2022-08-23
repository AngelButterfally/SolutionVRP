import copy
import sys
import os
import shutil
import time
import random
from tkinter import FALSE
from typing_extensions import Self
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sympy import ring
from VRPUI import Ui_MainWindow
import cv2 as cv
import numpy as np
import pandas as pd
import math as mt
from VRP import VRP_solution
from NetworkX import cal_shortest_path
from VRPTW import VRPTW_solution
from data import create_data_VRP

sys.path.append('./CVRP')


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.filename = './custumer.csv'
        self.customerpoint_list = None
        self.str1 = None
        self.car_number = 4
        self.Q = None
        self.Y = None
        self.X = None
        self.points = None
        self.change_point = None
        self.change_X = None
        self.change_Y = None
        self.change_Q = None
        self.car_distance = None
        self.VRP_data = None
        self.receive_point_list = []
        self.setupUi(self)

        # 展示地图
        self.label.setPixmap(QPixmap("./BIT_map.png"))  # 我的图片格式为png.与代码在同一目录下
        self.label.setScaledContents(True)  # 图片大小与label适应，否则图片可能显示不全
        # 引入所有坐标点
        self.All_points = pd.read_csv("./point2.csv", header=None)
        self.All_X = self.All_points.iloc[:, 1].values.T
        self.All_Y = self.All_points.iloc[:, 2].values.T
        # point2中引入的全坐标未处理，对其处理
        for m in range(len(self.All_points)):
            self.All_X[m] = mt.ceil(self.All_X[m] * 0.4)
            self.All_Y[m] = mt.ceil(self.All_Y[m] * 0.4)

        self.Custumer_Button.clicked.connect(self.showpoint)  # 输入坐标
        self.pushButton_auto_VRP.clicked.connect(self.VRP_auto)  # 自动VRP
        self.pushButton_manual_VRP.clicked.connect(self.VRP_mannual)  # 手动VRP

        self.ShowTW_Button.clicked.connect(self.VRPTW_show)  # VRP_TW训练展示
        self.CVRP_pushButton.clicked.connect(self.CVRP_show)  # CVRP训练展示
        self.TWcar_pushButton.clicked.connect(self.change_TWcar)  # 修改VRPTW车辆数目
        self.ChangeLocation_pushButton.clicked.connect(
            self.change_location)  # 修改目标点坐标
        self.ChangeQ_pushButton.clicked.connect(self.change_need)  # 修改目标点需求量
        self.choose_point_pushButton.clicked.connect(
            self.choose_receive_point)  # 选择接货点

    '''
    按键功能函数
    '''
    # 输入坐标

    def showpoint(self):
        self.choose_file()
        self.get_points()

    def VRP_auto(self):
        """自动VRP"""
        self.VRP_data = create_data_VRP(
            path=self.filename, vehicles_num=self.VRPcar_spinBox.value(), auto_random_select_point=True, auto_point_num=self.spinBox_auto_point_num.value())
        self.customerpoint_list = VRP_solution(self.VRP_data)  # 输入VRP训练结果
        self.new_showpic()
        self.cars_distance()

    def VRP_mannual(self):
        """手动VRP"""
        self.VRP_data = create_data_VRP(
            path=self.filename, vehicles_num=self.VRPcar_spinBox.value(), auto_random_select_point=FALSE, mannual_point_list=self.receive_point_list)
        self.customerpoint_list = VRP_solution(self.VRP_data)  # 输入VRP训练结果
        self.new_showpic()
        self.cars_distance()

    def VRPTW_show(self):
        """RP_TW训练展示"""
        self.customerpoint_list = VRPTW_solution(self.car_number)
        self.new_showpic()
        self.cars_distance()

    # CVRP训练展示
    def CVRP_show(self):
        from CVRP.api import run
        self.customerpoint_list = run(X=self.X, Y=self.Y, Q=self.Q)
        self.new_showpic()
        self.cars_distance()

    # 修改VRP_TW车辆数目
    def change_TWcar(self):
        change_car = self.TWcar_spinBox.value()
        self.car_number = change_car
        # print(self.carnumber)

    # 修改目标点坐标
    def change_location(self):
        self.change_data()
        self.restore()
        self.get_points()

    # 修改目标点需求量
    def change_need(self):
        self.change_q()
        self.restore()
        # self.get_points()

    def choose_receive_point(self):
        '''
        选择接货点
        '''
        self.receive_point_list.append(
            self.spinBox_mannual_point_index.value())
        print(self.receive_point_list)

    '''
    功能实现函数
    '''
    # 获取目标文件路径

    def choose_file(self):
        self.filename, filetype = QFileDialog.getOpenFileName(
            self, "选择文件", ".", "Text Files (*.csv)")

    # 从文件读取信息
    def get_points(self):
        if self.filename == '' or None:
            self.filename = './custumer.csv'
        self.points = pd.read_csv(self.filename, header=None)
        self.X = self.points.iloc[:, 0].values.T
        self.Y = self.points.iloc[:, 1].values.T
        # self.Q = np.empty(len(self.points), np.int64) # Q随机生成
        self.Q = self.points.iloc[:, 2].values.T    # Q直接读取

        # self.initial_X = copy.deepcopy(self.X)
        # self.initial_Y = copy.deepcopy(self.Y)

        for i in range(len(self.points)):
            # print(i)
            if i == 0:
                # self.Q[i] = 0
                self.str1 = " Form: X( " + str(self.X[i]) + " )" + \
                            ", Y( " + str(self.Y[i]) + " )\n\n"
            else:
                # self.Q[i] = random.randint(1, 20)
                self.str1 = self.str1 + " Customer" + str(i) + ": X( " + str(self.X[i]) + " )" + ", Y( " + str(
                    self.Y[i]) + " ), Q:" + str(self.Q[i]) + "\n\n"

        self.labellist.setWordWrap(True)
        self.labellist.setText(self.str1)
        self.labellist.setAlignment(Qt.AlignTop)
        # 将坐标点同图像适配
        for m in range(len(self.points)):
            self.X[m] = mt.ceil(self.X[m] * 0.4)
            self.Y[m] = mt.ceil(self.Y[m] * 0.4)

    # 对X,Y值进行修改
    def change_data(self):

        self.change_point = pd.read_csv('newpoint.csv', header=None)
        self.change_X = self.change_point.iloc[:, 0].values.T
        self.change_Y = self.change_point.iloc[:, 1].values.T

        Change_Location = self.spinBox_Number.value()
        New_Xvalue = self.spinBox_X.value()
        New_Yvalue = self.spinBox_Y.value()
        self.change_X[Change_Location] = New_Xvalue
        self.change_Y[Change_Location] = New_Yvalue

    # 对Q值进行修改
    def change_q(self):

        self.change_point = pd.read_csv('newpoint.csv', header=None)
        self.change_Q = self.change_point.iloc[:, 2].values.T

        Change_Location = self.spinBox_Number.value()
        New_Qvalue = self.spinBox_Q.value()
        self.change_Q[Change_Location] = New_Qvalue

    # 将修改过的数据重新存储到csv文件
    def restore(self):
        # summary = {'x': self.initial_X, 'y': self.initial_Y}

        summary = {'x': self.change_X, 'y': self.change_Y, 'Q': self.change_Q}
        df = pd.DataFrame(summary)
        df.to_csv('newpoint.csv', header=0, index=0)
        # df.to_csv(self.filename, header=0, index=0)

    # 综合输出完整路线画图方案
    def new_showpic(self):
        img = cv.imread('./BIT_map.png')
        if img is None:
            print('Failed to read map')
            sys.exit()
        else:
            # 将custumer点标记出来
            for i in range(len(self.points)):
                img = cv.circle(
                    img, (self.X[i] + 3, self.Y[i] + 3), 4, (0, 255, 0), -1)

            for n in range(len(self.customerpoint_list)):
                # bgr = np.random.randint(0, 255, 3, dtype=np.int32)  # 对每辆车生成随机颜色
                bgr = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.897, 0.187, 0.704], [0.316, 0.463, 0.931],
                       [0.255, 0.675, 0.116],
                       [0.0627, 0.596, 0.933], [0.498, 0.0235, 0.980], [0.98, 0.0235, 0.451], [0.0196, 0.98, 0.584]]
                for m in range(len(self.customerpoint_list[n]) - 1):
                    # 计算获得每两个收货点最短路线
                    point_forward = self.customerpoint_list[n][m]
                    point_backword = self.customerpoint_list[n][m + 1]
                    a = cal_shortest_path(
                        int(point_forward), int(point_backword))
                    # print(a[0])

                    # 画出每两个点之间的运动轨迹
                    for i in range(len(a[0]) - 1):
                        list_before = a[0][i]
                        list_next = a[0][i + 1]
                        img = cv.arrowedLine(img, (self.All_X[list_before] + 3, self.All_Y[list_before] + 3),
                                             (self.All_X[list_next] + 3,
                                              self.All_Y[list_next] +
                                              3),
                                             (255 * bgr[n][0], 255 * bgr[n][1], 255 * bgr[n][2]), 3, cv.LINE_8, 0, 0.1)
                        # img = cv.line(img, (self.All_X[list_before] + 3, self.All_Y[list_before] + 3),
                        #               (self.All_X[list_next] + 3, self.All_Y[list_next] + 3),
                        #               (255 * bgr[n][0], 255 * bgr[n][1], 255 * bgr[n][2]), 3, cv.LINE_8, 0)
                        # img = cv.line(img, (self.All_X[list_before] + 3, self.All_Y[list_before] + 3),
                        #               (self.All_X[list_next] + 3, self.All_Y[list_next] + 3),
                        #               (np.int(bgr[0]), np.int(bgr[1]), np.int(bgr[2])), 3, cv.LINE_8, 0)
                        cv.imwrite('./final/BIT_map_' + str(n) +
                                   '_' + str(m) + '_' + str(i) + '.jpg', img)
                        self.label.setPixmap(
                            QPixmap("./final/BIT_map_" + str(n) + "_" + str(m) + '_' + str(
                                i) + ".jpg"))  # 我的图片格式为png.与代码在同一目录下
                        self.label.setScaledContents(
                            True)  # 图片大小与label适应，否则图片可能显示不全
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.05)

                        if not os.path.exists("./final"):
                            os.mkdir("./final")
                        else:
                            shutil.rmtree("./final")
                            os.mkdir("./final")

    # 输出各车辆运行距离
    def cars_distance(self):
        str2 = str()
        b = np.zeros(len(self.customerpoint_list))
        total_distance = 0
        for n in range(len(self.customerpoint_list)):
            for m in range(len(self.customerpoint_list[n]) - 1):
                # 计算获得每两个收货点最短路线
                point_forward = self.customerpoint_list[n][m]
                point_backword = self.customerpoint_list[n][m + 1]
                a = cal_shortest_path(int(point_forward), int(point_backword))
                b[n] = a[1]
        self.car_distance = b

        for m in range(0, len(self.car_distance)):
            str2 = str2 + "第" + (str(m + 1)) + "辆车的运行距离：" + \
                str(mt.ceil(self.car_distance[m])) + "\n"
            total_distance = mt.ceil(self.car_distance[m]) + total_distance
            # print(total_distance)
            if m == (len(self.car_distance) - 1):
                str2 = str2 + "总运行距离：" + str(total_distance) + "\n"
                self.str1 = str2 + "\n" + self.str1

        self.labellist.setWordWrap(True)
        self.labellist.setText(self.str1)
        self.labellist.setAlignment(Qt.AlignTop)

    # 清空文件夹内文件
    def RemoveDir(self, filepath):
        # 如果文件夹不存在就创建，如果文件存在就清空！
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath)
            os.mkdir(filepath)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    # myWin.showFullScreen()
    myWin.showMaximized()
    myWin.show()
    sys.exit(app.exec_())
