#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import numpy as np
import cv2
import sys

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)
		self._template_path = 0
		self._cam_id = 0
		self._cam_fps = 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self._template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self._template_path)
		self.template_label.setPixmap(pixmap)
		print("Loaded template image file: " + self._template_path)

	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		#TODO run SIFT on the captured frame
		ref = cv2.imread(self._template_path) #load query image to compare against
		sift = cv2.xfeatures2d.SIFT_create()

		#find keypoints of query image
		gref = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
		kp_gref, desc_gref = sift.detectAndCompute(gref,None)
		cv2.drawKeypoints(gref, kp_gref,gref)

		#find keypoints from live-feed
		gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		kp_gframe, desc_gframe = sift.detectAndCompute(gframe,None)
		cv2.drawKeypoints(gframe, kp_gframe,gframe)
		
		#SIFT magic
		index_params = dict(algorithm=0,trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(desc_gref,desc_gframe,k=2)

		good_points = [m for m,n in matches if m.distance < 0.6*n.distance]

		#draw matches between query and live feed
		matched_frame = cv2.drawMatches(gref,kp_gref,frame,kp_gframe,good_points,None)

		#homography
		if len(good_points) > 6:
			query_pts = np.float32([kp_gref[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
			train_pts = np.float32([kp_gframe[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

			matrix,mask = cv2.findHomography(query_pts,train_pts,cv2.RANSAC,5.0)

			matches_mask = mask.ravel().tolist()

			#perspective transform
			h,w = gref.shape

			pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

			dst = cv2.perspectiveTransform(pts,matrix)

			homography = cv2.polylines(frame,[np.int32(dst)],True,(255,0,0),3)
			pixmap_homography = self.convert_cv_to_pixmap(homography)
			self.live_image_label.setPixmap(pixmap_homography)

		else:
			pixmap_matches = self.convert_cv_to_pixmap(matched_frame)
			self.live_image_label.setPixmap(pixmap_matches)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
