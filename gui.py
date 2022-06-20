import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import tensorflowModel


class Ui_MainWindow(object):
    
    #variables needed to work in the program
    folderPath = ''
    model = None
    textLog = "App was opened\n"
    imagePath = "./ds/single_test/0.jpg"

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("TestImagePython")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.textPanel = QtWidgets.QTextBrowser(self.centralwidget)
        self.textPanel.setGeometry(QtCore.QRect(20, 20, 301, 401))
        self.textPanel.setObjectName("textPanel")

        self.widgetPhoto = QtWidgets.QWidget(self.centralwidget)
        self.widgetPhoto.setGeometry(QtCore.QRect(330, 70, 300, 180))
        self.widgetPhoto.setObjectName("widgetPhoto")
        self.Photo = QtWidgets.QLabel(self.widgetPhoto)
        self.Photo.setGeometry(QtCore.QRect(0, 0, 300, 180))
        self.Photo.setObjectName("Photo")
        self.Photo.setPixmap(QtGui.QPixmap(self.imagePath))
        self.Photo.setScaledContents(True)
        self.Photo.setText("")
        self.createModel = QtWidgets.QPushButton(self.centralwidget)
        self.createModel.setGeometry(QtCore.QRect(360, 310, 101, 41))
        self.createModel.setObjectName("createModel")
        self.testModel = QtWidgets.QPushButton(self.centralwidget)
        self.testModel.setGeometry(QtCore.QRect(500, 310, 101, 41))
        self.testModel.setObjectName("testModel")
        self.testCurrentImage = QtWidgets.QPushButton(self.centralwidget)
        self.testCurrentImage.setGeometry(QtCore.QRect(500, 370, 101, 41))
        self.testCurrentImage.setObjectName("testCurrentImage")
        self.loadBest = QtWidgets.QPushButton(self.centralwidget)
        self.loadBest.setGeometry(QtCore.QRect(360, 370, 101, 41))
        self.loadBest.setObjectName("loadBest")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(330, 20, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.title.setFont(font)
        self.title.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAdd_Image = QtWidgets.QAction(MainWindow)
        self.actionAdd_Image.setObjectName("actionAdd_Image")
        self.actionAdd_Model = QtWidgets.QAction(MainWindow)
        self.actionAdd_Model.setObjectName("actionAdd_Model")
        self.menuFile.addAction(self.actionAdd_Image)
        self.menuFile.addAction(self.actionAdd_Model)
        self.menubar.addAction(self.menuFile.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #register a button click
        self.actionAdd_Image.triggered.connect(lambda: self.clickAddImg(self.widgetPhoto))
        self.actionAdd_Model.triggered.connect(lambda: self.clickLoadModel(self.widgetPhoto))
        self.loadBest.clicked.connect(self.clickLoadBestModel)
        self.createModel.clicked.connect(self.clickCreateModel)
        self.testModel.clicked.connect(self.clickTestModel)
        self.testCurrentImage.clicked.connect(self.clickTestCurrentImage)

    #print text to text panel
    def insertToTextLog(self,text):
        self.textLog += (text + "\n")
        self.textPanel.setText(self.textLog)

    #test current image after a click
    def clickTestCurrentImage(self):
        if not (self.model):
            self.insertToTextLog("Model doesn't exist yet")
            return

        #print predicted answer
        self.insertToTextLog(tensorflowModel.testImage(self.model,self.imagePath))
        
        #print correct answer
        startIndex = self.imagePath.rfind("/")
        data = pd.read_csv('./csv/test.csv')
        for img in (data.values):
            if img[0] == self.imagePath[startIndex+1:]:
                self.insertToTextLog("In the picture you can see: " + img[1])
           
    #Tests a loaded model after a click of a button
    def clickTestModel(self):
        if not (self.model):
            return
        self.insertToTextLog(tensorflowModel.testModel(self.model))

    #creates a model with a click (takes some time)
    def clickCreateModel(self):
        self.model = tensorflowModel.createModel()
        self.insertToTextLog("Created new model")

    #loades the best model from a path
    def clickLoadBestModel(self):
        self.folderPath = './models/best_model/'
        self.model = tensorflowModel.loadModel(self.folderPath)
        self.insertToTextLog("Loaded Best model")

    #loades a model choosen by the user
    def clickLoadModel(self,widgetPhoto):
        
        #prompt to choose folder
        self.folderPath = QFileDialog.getExistingDirectory(widgetPhoto, "Select Directory",'./models/')
        if not (self.folderPath):
            return

        #loading a model from a given path
        self.model = tensorflowModel.loadModel(self.folderPath)
        self.insertToTextLog("Loaded new model")
        self.model.summary()

    #adding a image with a button click
    def clickAddImg(self,widgetPhoto):
        #open a file from os dir
        fname = QFileDialog.getOpenFileName(widgetPhoto, 'Open Image','./ds/single_test/', "Image files (*.jpg)")
        
        #check if the img was choosen and exists
        if not (fname) or fname[0] == '':
            return
        
        #add image to path and display it
        self.imagePath = fname[0]
        pixmap = QPixmap(self.imagePath)
        pixmap = pixmap.scaled(300,180,1)
        self.Photo.setPixmap(pixmap)
        self.insertToTextLog("Added new image " + self.imagePath)

    #names of buttons
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        MainWindow.setWindowIcon(QtGui.QIcon("icon.webp"))

        self.createModel.setText(_translate("MainWindow", "Create new Model"))
        self.testModel.setText(_translate("MainWindow", "Test Model"))
        self.testModel.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.testCurrentImage.setText(_translate("MainWindow", "Test current image"))
        self.loadBest.setText(_translate("MainWindow", "Load Best Model"))
        self.title.setText(_translate("MainWindow", "Check if its a whale or a dolphin"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionAdd_Image.setText(_translate("MainWindow", "Add Image"))
        self.actionAdd_Image.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionAdd_Model.setText(_translate("MainWindow", "Load Model"))
        self.actionAdd_Model.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.textPanel.setText(_translate('MainWindow',self.textLog))

#main loop
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
