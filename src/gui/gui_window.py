import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import json
from pprint import pprint
import math
import sip
import os

from core import default

from gui import gui_api

class Spoiler:
    def __init__(self, title, start, parent = None):
        self.widget = QWidget()
        self.toggleButton = QToolButton()
        self.toggleButton.setStyleSheet("QToolButton { border: none; }")
        self.toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggleButton.setArrowType(QtCore.Qt.RightArrow)
        self.toggleButton.setText(title)
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(start)
        
        self.headerLine = QFrame()
        self.headerLine.setFrameShape(QFrame.HLine)
        self.headerLine.setFrameShadow(QFrame.Sunken)
        self.headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
        # don't waste space
        self.mainLayout = QGridLayout()
        self.mainLayout.setVerticalSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0;
        self.mainLayout.addWidget(self.toggleButton, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.mainLayout.addWidget(self.headerLine, 0, 2, 1, 1)
        self.widget.setLayout(self.mainLayout)
        
        self.frame = QWidget(parent)
        
        _this = self
        
        def click(checked):
            if not checked:
                _this.frame.show()
            else:
                _this.frame.hide()
            _this.toggleButton.setArrowType(QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow)
        self.toggleButton.clicked.connect(click)
        
        click(start)


class Slider:
    def __init__(self, param, func):
        self.name = param["name"]
        self.func = func
        
        self.width = 10.
        
        self.log = "scale" in param and param["scale"] == "logarithmic"
        if "scale" in param and param["scale"] == "int":
            self.width = 1
        
        self.slider = QSlider(QtCore.Qt.Horizontal)
        
        self.slider.valueChanged.connect(lambda x : self.valueChange(x))
        
        self.label = QLabel()
        
        self.slider.setValue(1)
        
        if self.log:
            self.slider.setRange(math.log10(param["min"]) * self.width, math.log10(param["max"]) * self.width)
            self.slider.setValue(math.log10(param["default"]) * self.width if ("default" in param) else (math.log10(param["min"]) + math.log10(param["max"])) * self.width/2.0)
        else:
            self.slider.setRange(param["min"] * self.width, param["max"] * self.width)
            self.slider.setValue(param["default"] * self.width if ("default" in param) else (param["min"] + param["max"]) * self.width/2.0)
        
    
    def valueChange(self, x):
        if self.log:
            self.label.setText(self.name + " : " + str('{:0.1e}'.format(10**(x/self.width))).rjust(4))
            self.func(10**(x/self.width))
        else:
            self.label.setText(self.name + " : " + str(x/self.width).rjust(4))
            self.func(x/self.width)
        
class Gui:
    def __init__(self, parent = None):

        with open(os.path.dirname(__file__) + '/param_list.json') as f:
            self.params = json.load(f)
            
        st = QWidget(parent)
        prime = QVBoxLayout(st)
        st.setLayout(prime)
        prime.setContentsMargins(0, 0, 0, 0)
        
        #gui
        self.scroll = QScrollArea(st)
        prime.addWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        
        self.scrollInner = QWidget()
        
        policy = self.scrollInner.sizePolicy()

        policy.setVerticalStretch(0);   
        policy.setHorizontalStretch(1);
        
        policy.setVerticalPolicy(4)
        
        self.scrollInner.setSizePolicy(policy)
        
        self.form = QVBoxLayout(self.scrollInner)
        self.scroll.setWidget(self.scrollInner)
        
        self.form.setContentsMargins(10, 10, 10, 10)
        
        self.scroll.resize(300,800)
        
        
        title = QLabel("Parameters")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QFont("Arial", 15, QtGui.QFont.Bold));
        self.form.addWidget(title)
        
        self.prepareGui()
        
        def run():
            gui_api.call(self.params["function_name"], self.values)
        
        runbtn = QPushButton("Run", parent)
        prime.addWidget(runbtn)
        runbtn.resize(200, 300)
        
        runbtn.pressed.connect(run)
        
        
    def prepareGui(self):
        current = self.form
        
        self.values = {}
        
        for group in self.params["parameter-groups"]:
            
            if "multiple" in group:
                widget = QPushButton("Add new " + group["name"])
                group["index"] = 0
                self.values[group["name"]] = []
                
                def openNew(_group=group, i=self.form.count()):
                    _group["index"] += 1
                    spoiler = self.addGroup(_group, _group["index"])
                    self.form.insertWidget(i, spoiler.widget)
                    self.form.insertWidget(i+1, spoiler.frame)
                widget.pressed.connect(openNew)
                
            
            spoiler = self.addGroup(group)
            
            self.form.addWidget(spoiler.widget)
            self.form.addWidget(spoiler.frame)
            
            if "multiple" in group:
                self.form.addWidget(widget)
            
            
            
    
    def addGroup(self, group, i=0):
        
        if "multiple" in group:
            self.values[group["name"]].append({})
            
        else:
            self.values[group["name"]] = {}
        
        current = QVBoxLayout()
        widget = Spoiler(group["name"], "optional" in group, self.scrollInner)
        widget.frame.setLayout(current)
        
        for param in group["parameters"]:
            
            if "multiple" in group:
                def updateValue(value, _param=param["name"], _group=group["name"], i=i):
                    print(_group + "  " + str(i) + "  " + _param + "  " + str(value))
                    self.values[_group][i][_param] = value
                    
                self.addWidget(param, current, updateValue)
            else:
                def updateValue(value, _param=param, _group=group):
                    print(_group["name"] + "  " + _param["name"] + "  " + str(value))
                    self.values[_group["name"]][_param["name"]] = value
                
                self.addWidget(param, current, updateValue)
            
            
        if "multiple" in group:
            delbutton = QPushButton("Remove this " + group["name"])
            def remove(rem=widget, _group=group["name"], i=i):
                self.values[_group][i] = None
                self.form.removeWidget(rem.frame)
                self.form.removeWidget(rem.widget)
                sip.delete(rem.frame)
                sip.delete(rem.widget)
                
            delbutton.pressed.connect(remove)
            current.addWidget(delbutton)
        
        return widget
        
        
    def addWidget(self, param, current, updateValue):
        
        try:
            param["default"] =  getattr(globals()["default"], param["name"])
        except:
            pass
        
        if param["type"] == "slider":
            
            slider = Slider(param, updateValue)
            current.addWidget(slider.label)
            current.addWidget(slider.slider)
            
        elif param["type"] == "selector":
            selector = QComboBox()
            
            def update(x, func=updateValue, vals=param["values"]):
                func(vals[x])
            selector.currentIndexChanged.connect(update)
            
            for i in range(len(param["values"])):
                selector.insertItem(i, str(param["values"][i]))
            
            pprint(param)
            selector.setCurrentIndex(param["values"].index(param["default"]))
            
            current.addWidget(QLabel(param["name"] + " : "))
            current.addWidget(selector)
            
        elif param["type"] == "toggle":
            
            toggle = QPushButton(param["name"])
            toggle.setCheckable(True)
            toggle.toggled.connect(updateValue)
            toggle.toggle()
            toggle.setDown(param["default"])
            current.addWidget(toggle)
            
        elif param["type"] == "number":
            
            widget = QLineEdit()
            widget.setValidator(QIntValidator())
            def update(x, func=updateValue):
                func(float(x))
            widget.textChanged.connect(update)
            current.addWidget(QLabel(param["name"] + " : "))
            current.addWidget(widget)
            widget.setText(str(param["default"]))
            
        elif param["type"] == "file":
            
            widget = QPushButton(param["name"])
            def openDialog(i=0, func=updateValue):
                func(QFileDialog.getOpenFileName(None, "Choose File", ""))
            widget.pressed.connect(openDialog)
            current.addWidget(widget)
            
            updateValue("")
            
        elif param["type"] == "files":
            
            widget = QPushButton(param["name"])
            def openDialog(i=0, func=updateValue):
                func(QFileDialog.getOpenFileNames(None, "Choose File", ""))
            widget.pressed.connect(openDialog)
            current.addWidget(widget)
            
            updateValue([])

class View:
    def __init__(self, parent = None):
        self.qvtk = QVTKRenderWindowInteractor(parent)

        self.ren = vtk.vtkRenderer()
        self.qvtk.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.qvtk.GetRenderWindow().GetInteractor()

        # Create a float array which represents the points.
        pcoords = vtk.vtkFloatArray()
        # Note that by default, an array has 1 component.
        # We have to change it to 3 for points
        pcoords.SetNumberOfComponents(3)
        # We ask pcoords to allocate room for at least 4 tuples
        # and set the number of tuples to 4.
        pcoords.SetNumberOfTuples(4)
        # Assign each tuple. There are 5 specialized versions of SetTuple:
        # SetTuple1 SetTuple2 SetTuple3 SetTuple4 SetTuple9
        # These take 1, 2, 3, 4 and 9 components respectively.
        pcoords.SetTuple3(0, 0.0, 0.0, 0.0)
        pcoords.SetTuple3(1, 0.0, 1.0, 0.0)
        pcoords.SetTuple3(2, 1.0, 0.0, 0.0)
        pcoords.SetTuple3(3, 1.0, 1.0, 0.0)

        # Create vtkPoints and assign pcoords as the internal data array.
        points = vtk.vtkPoints()
        points.SetData(pcoords)

        # Create the cells. In this case, a triangle strip with 2 triangles
        # (which can be represented by 4 points)
        strips = vtk.vtkCellArray()
        strips.InsertNextCell(4)
        strips.InsertCellPoint(0)
        strips.InsertCellPoint(1)
        strips.InsertCellPoint(2)
        strips.InsertCellPoint(3)

        # Create an integer array with 4 tuples. Note that when using
        # InsertNextValue (or InsertNextTuple1 which is equivalent in
        # this situation), the array will expand automatically
        temperature = vtk.vtkIntArray()
        temperature.SetName("Temperature")
        temperature.InsertNextValue(10)
        temperature.InsertNextValue(20)
        temperature.InsertNextValue(30)
        temperature.InsertNextValue(40)

        # Create a double array.
        vorticity = vtk.vtkDoubleArray()
        vorticity.SetName("Vorticity")
        vorticity.InsertNextValue(2.7)
        vorticity.InsertNextValue(4.1)
        vorticity.InsertNextValue(5.3)
        vorticity.InsertNextValue(3.4)

        # Create the dataset. In this case, we create a vtkPolyData
        polydata = vtk.vtkPolyData()
        # Assign points and cells
        polydata.SetPoints(points)
        polydata.SetStrips(strips)
        # Assign scalars
        polydata.GetPointData().SetScalars(temperature)
        # Add the vorticity array. In this example, this field
        # is not used.
        polydata.GetPointData().AddArray(vorticity)

        # Create the mapper and set the appropriate scalar range
        # (default is (0,1)
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(polydata)
        self.mapper.SetScalarRange(0, 40)

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.ren.AddActor(self.actor)

        self.ren.ResetCamera()
        
        self.qvtk.resize(800, 800)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Deformetrica")
    
    bar = window.menuBar()
    file = bar.addMenu("File")
    save = file.addAction("Run")
    edit = bar.addMenu("Edit")
    view = bar.addMenu("View")
    
    split = QSplitter()
    window.setCentralWidget(split)
    
    gui = Gui(split)
    
    
    view = View(split)
    view.qvtk.AddObserver("ExitEvent", lambda o, e, a=app: a.quit())
    
    
    window.show()
    
    window.resize(1200, 900)
    
    view.iren.Initialize()
    view.iren.Start()
    
    app.exec_()
