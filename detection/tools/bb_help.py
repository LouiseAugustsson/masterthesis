import cv2
import copy as c
import numpy as np


def check_pedestrian_height(annotation_file):
	large_enough = False
	with open(annotation_file) as a:
		heigth = []
		occ = []
		for line in a:
			if line[0:7] == 'person ':
				line = line.split()
				if int(line[4])>50 and int(line[5]) == 0:
					large_enough = True
					heigth.append(int(line[4]))
					occ.append(int(line[5]))
	return heigth, large_enough, occ


def read_annotations(annotation_file):
	#print annotation_file
	left_corners_person = []
	right_corners_person = []
	left_corners_people = []
	right_corners_people = []
	with open(annotation_file, 'r') as a:
		for line in a:
			if line[0:7] == 'person ':
				line =  line.split()
				if int(line[4])>50 and int(line[5]) == 0: #No occlusion
					left_corners_person.append([int(line[1]), int(line[2])])
					right_corners_person.append([int(line[1])+int(line[3]), int(line[2])+int(line[4])])
			elif line[0:3] == 'peo':
				line =  line.split()
				left_corners_people.append([int(line[1]), int(line[2])])
				right_corners_people.append([int(line[1])+int(line[3]), int(line[2])+int(line[4])]) 

	return left_corners_person, right_corners_person, left_corners_people, right_corners_people

def find_fitted_bounding_boxes(left_corner, right_corner):
	#upper left corner, lower right corner.
	#new_left = np.zeros(2, dtype = int)
	new_left = []
	new_left.append(int(round(((left_corner[0]-8) / 16.))*16+8))
	new_left.append(int(round(((left_corner[1]-8) / 16.))*16+8))
	new_right = []
	new_right.append(int(round(((right_corner[0]-8) / 16.))*16+8))
	new_right.append(int(round(((right_corner[1]-8) / 16.))*16+8))
	return new_left, new_right

def plot_grid(image):
	#plot grid on image of size 640 x 480. Box size; 16 x 16. Discards 8 pixels at each side.
	for i in range(0,40):
		line_pos = 8 + i*16
		if i <= 30:
			cv2.line(image, (8,line_pos), (632,line_pos), (0,0,255), 1)
		cv2.line(image,(line_pos, 8), (line_pos,472), (0,0,255), 1)	

	return image

def plot_bounding_boxes(image, left_corner, right_corner, color, size):
	cv2.rectangle(image, (left_corner[0], left_corner[1]), (right_corner[0], right_corner[1]), color, size)
	return 

def find_label_matrix(left_corners, right_corners, label_matrix):
	for i in range(0,len(left_corners)):
		left = (left_corners[0]-8)/16
		right = (right_corners[0]-8)/16
		top = (left_corners[1]-8)/16
		bottom = (right_corners[1]-8)/16
		for j in range(top,bottom):
			for k in range(left,right):
				if j !=29 and k != 39:
					label_matrix[j,k] = 1
		return label_matrix


