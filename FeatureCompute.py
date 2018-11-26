'''
Created on 2018-11-12

@author: Zhaoyu Sun
'''


import cv2 as cv
import numpy as np
import os
import datetime

class FeatureCompute(object):

	'''
	A class generate feature vector for image or subimage by SIFT and BOW.

	Each time you assign change the number of SIFT feature points or other
	params of BOW,you need to run initialize function firstly.

	An advice, the number of SIFT feature points should not be less than 120,
	and for BOW, it is a K-Means aggregate essentaly, so be sure you are
	familiar with K-Means.

	I design a method to offer you training data, both positive and negative.
	I also implement a method to generate a feature vector for each image or
	subimage based on you vocabulary.

	Except initialize, generatePhi, generateTrainData, other functions are all
	private, please don't modify it by yourself. If you get new idea or want to
	do some optimization, please inform me firstly.If you want to nosetest some
	function, please implement another test file on your server.

	Var:
		num_phi: # of SIFT feature points
		wordCnt: dimension of final output
		iterTime: parameter for cv.Kmeans,iteration times
		explosion: parameter for cv.Kmeans,tolerance
	'''

	def __init__(self, num_phi = 300,
				 wordCnt=70, iterTime=50, explosion=0.01):
		self.dir = os.getcwd()
		self.num_phi = num_phi
		self.wordCnt = wordCnt
		self.iterTime = iterTime
		self.explosion = explosion

	def _calcSiftFeature(self, img):
		'''
		Method to compute SIFT features.
		Params:
			img: ndarray
		Returns:
			n * 128 ndarray, n equals to num_phi
		'''
		#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		sift = cv.xfeatures2d_SIFT.create(self.num_phi)
		kp, des = sift.detectAndCompute(img, None)
		return des

	def _initFeatureSet(self):
		'''
		Method to initialize SIFT features of all images, the features are
		stored in /features/phi.npy
		Params:
			None
		Returns:
			None
		'''
		name = (str(self.wordCnt) + '_' + str(self.iterTime) + '_' +
			    str(self.explosion))
		featureSet = np.float32([]).reshape(0,128)
		print("Extract features from TrainSet :\n")
		num_img = len([img for img in os.listdir(self.dir + '/Image')]) - 1
		for count in range(num_img):
			filename = self.dir + "/Image/" + str(count) + '.jpg'
			img = cv.imread(filename)
			des = self._calcSiftFeature(img)
			featureSet = np.append(featureSet, des, axis=0)
			featCnt = featureSet.shape[0]
		print(str(featCnt) + " features in "
			+ str(num_img) + " images\n")
		# save featureSet to file
		filename = self.dir + "/Features/" + name + "_phi.npy"
		np.save(filename, featureSet)
		print("Finsh Initializing Phi")

	def _learnVocabulary(self):
		'''
		Method to Compute Kmeans Centers and implement BOW, Centers and Lables
		are stored in /vocabulary/bow.npy
		Params:
			None
		Returns:
			None
		'''
		name = (str(self.wordCnt) + '_' + str(self.iterTime) + '_' +
			    str(self.explosion))
		filename = self.dir + "/Features/" + name + "_phi.npy"
		try:
			features = np.load(filename)
		except Exception as e:
			print("SIFT feature file doesn't exist!!!")
		print("Learn vocabulary ...")
		# use k-means to implement BOW
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
					self.iterTime, self.explosion)
		flags = cv.KMEANS_RANDOM_CENTERS
		# PP_Center??
		compactness, labels, centers = cv.kmeans(features, self.wordCnt, None,
												 criteria, self.iterTime,
												 flags)
		filename = self.dir + "/Vocabulary/" + name + "_bow.npy"
		np.save(filename, (labels, centers))
		print("Finish BOW\n")

	def _calcFeatVec(self, features, centers):
		'''
		Method to compute feature vector for each image/subimage
		Params:
			None
		Return:
			1 * wordCnt vector
		'''
		# feature: sift feature, m * 128 vector, m = # of feature points
		# centers: k-means centers
		featVec = np.zeros((1, self.wordCnt))
		for i in range(0, features.shape[0]):
			fi = features[i]
			diffMat = np.tile(fi, (self.wordCnt, 1)) - centers
			sqSum = (diffMat**2).sum(axis=1)
			dist = sqSum**0.5
			sortedIndices = dist.argsort()
			idx = sortedIndices[0] # index of the nearest center
			featVec[0][idx] += 1
		return featVec

	def generateTrainData(self):
		'''
			Function to generate train data to train your classifier
			Params:
				num_pos: # of positive samples
				num_neg: # of negative samples
			Returns:
				Tuple: (trainData, Lables)
				trainData: wordCnt * m matrix, m: # of trainging images
				Labels: m * 1 matrix, m: # of trainging images
		'''
		name = (str(self.wordCnt) + '_' + str(self.iterTime) + '_' +
			    str(self.explosion))
		trainData = np.float32([]).reshape(0,self.wordCnt)
		response = np.float32([])
		try:
			labels, centers = np.load(self.dir + "/Vocabulary/" +
									name + "_bow.npy")
		except Exception as e:
			print("No Vocabulary file!!")
		num_pos = len([img for img in os.listdir(self.dir +
												'/PositiveSample')]) - 1
		for count in range(num_pos):
			filename = (self.dir + "/PositiveSample/" + str(count) + '.jpg')
			print(filename)
			img = cv.imread(filename)
			features = self._calcSiftFeature(img)
			featVec = self._calcFeatVec(features, centers)
			trainData = np.append(trainData, featVec, axis=0)
		pos_label = np.repeat(np.float32([1]), count + 1)
		response = np.append(response, pos_label)
		num_neg = len([img for img in os.listdir(self.dir +
												'/NegativeSample')]) - 1
		for count in range(num_neg):
			filename = (self.dir + "/NegativeSample/" + str(count) + '.jpg')
			print(filename)
			img = cv.imread(filename)
			features = self._calcSiftFeature(img)
			featVec = self._calcFeatVec(features, centers)
			trainData = np.append(trainData, featVec, axis=0)
		neg_label = np.repeat(np.float32([0]),count + 1)
		response = np.append(response, neg_label)
		response.reshape(-1, 1)
		return trainData.T, response

	def initialize(self):
		'''
			Function to initialize and create file
			Run this Function Firstly before you train your classifier!

			Params:
				None
			Returns:
				None
		'''
		self._initFeatureSet()
		self._learnVocabulary()

	def generatePhi(self, img):
		'''
			Function to generate 1 * wordCnt feature vector for any image or
			subimage
			Params:
				image
			Returns:
				wordCnt * 1 feature vector
		'''
		name = (str(self.wordCnt) + '_' + str(self.iterTime) + '_' +
			    str(self.explosion))
		features = self._calcSiftFeature(img)
		try:
			labels, centers = np.load(self.dir + "/Vocabulary/" +
									  name + "_bow.npy")
		except Exception as e:
			print("No Vocabulary file!!")
		featVec = self._calcFeatVec(features, centers)
		return featVec.T

if __name__ == '__main__':
	generate = FeatureCompute()
	generate.initialize()
	x, y = generate.generateTrainData()
	print(x.shape)
	print(y.shape)
	np.save(('TrainData/TrData' +
			datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			+ '.npy'), x)
	np.save(('TrainData/TrLabel' +
			datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			+ '.npy'), y)
