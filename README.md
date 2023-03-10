# CV_DataCleaner
When downloading images from the Internet using a scraper like bing_image_downloader, there is a problem of strong data noise. This set is designed to solve this problem.

<br><strong>Sequential launch of utilities to get the final result:</strong><br>

<strong>run.py</strong><br> 
Creates a directory ready for deletion and transferring files into it that the algorithm considers unnecessary. The created intermediate dictionaries, if any, are also transferred there.<br>
Example:<br>
--directory /content/images/sea<br>
--image Image_1.jpg<br>
--distance 1.1<br>
--out /content/images/features<br>

for details -h

<br><strong>To run individually:</strong><br>

<strong>get_futures.py</strong>  
Creates a dictionary file containing the names of image files and their corresponding features extracted from them
It accepts the address of the directory with images and the address of the directory where the file should be saved. If no out directory is specified, saves the dictionary file to the directory where the pictures are located.<br>
Example:<br>
--directory /content/images/sea<br>
--out /content/images/features<br>

for details -h

<strong>get_same.py</strong><br> 
Creates a dictionary with a list of image addresses that do not match the pattern.
It accepts the name of the image to be compared with, the address of the dictionary file containing the features, the address of the directory where the file should be saved and the distance in the distribution field - the maximum distance from the sample to the extreme representation of the image.<br>
Example:<br>
--image Image_1.jpg<br>
--directory /content/images/sea<br>
--out /content/images/features<br>
--distance 1.1<br>

When accessed as a function, it simply returns a dictionary, without saving.<br>

for details -h

<strong>requirements.txt</strong><br> 
Contains an indication of the required libraries.<br><br>

<strong>Version 0.1. The repository will be updated<strong>
