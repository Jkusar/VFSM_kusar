# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:36:36 2023

@author: Jernej Kušar
"""
import sys
import numpy as np
import glob
import os
import cv2
import string
from PIL import Image, ImageFilter 
from pathlib import Path
import io
import shutil
import re
from screeninfo import get_monitors
import lvm_read

#send email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

#progress bar
from tqdm import tqdm


def convert_mts_to_mp4(input_file, output_file, error_path):
    """
    Converts MTS files to mp4 format. 
    
    Input parameteters:
        - input_file [str] - path to the MTS video that you wish to convert
        - output_file [str] - path ti the mp4 video file that you wish to save
    """
    video_name = os.path.basename(input_file)
    try:
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print("Error: Unable to open input file.")
            return

        fps, frame_size = int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_file, 0, fps, frame_size)  # Use 0 for codec

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

            frame_count += 1
            print(f"Converting: {frame_count / total_frames * 100:.2f}% completed", end='\r')
        print(f"\nConversion for {video_name} completed.")
        
        cap.release()
        out.release()

    except Exception as e:
        err = open(error_path, mode = "a")
        p = os.path.basename(input_file)
        err.write(f"Error report for {p}. Exception: {e}\n")
        err.close()
        print(f"Conversion for {video_name} failed: {e}")
        
def crop_video(input_path, output_path, x, y, width, height, video_name=None):
    """
    Crops a video
    
    Input parameters:
        - input_path [str] - path to the video file you wish to crop
        - output_path [str] - path to the folder you wish to save the croped video to
        - x [int] - X position of origin from which croping values are defined (in px)
        - y [int] - y position of origin from which croping values are defined (in px)
        - height [int] - height of the croped video (in px)
        - width [int] - width of the croped video (in px)
        - video_name [str, optional] - name of the croped video. By default name does not change
    """
    
    if video_name is not None:
        output_path = output_path + "/" + video_name
    else:
        output_path = output_path + "/" + os.path.basename(input_path)
        
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the frames per second (fps) and frame dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if cropping parameters are within bounds
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height: 
        print(f"Cropping parameters are out of bounds.\n Parameters: \n  x = {x} > 0 \n y = {y} > 0 \n x + width  = {x+width} < {frame_width} \n y + height = {y+height} < {frame_height}")
        return
    
    # Define the codec and create VideoWriter object to save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    while True:
        frame_count += 1 
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame based on the specified coordinates and dimensions
        cropped_frame = frame[y:y+height, x:x+width]

        # Write the cropped frame to the output video
        out.write(cropped_frame)
        
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.2f}%")
        
    print(f"Video {os.path.basename(input_path)} cropped sucesfully.")
    
    cap.release()
    out.release()

def pil_to_opencv(pil_image):
    """
    Convert a PIL Image to an OpenCV-compatible format (NumPy array).
    """
    # Convert from RGB (PIL format) to BGR (OpenCV format)
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image
       
def frames_to_video(frames, output_path, fps, video_name=None, x=0, y=0, width=None, height=None, play_video=False):
    """
    Crops a video
    
    Input parameters:
        - frames [list] - input list of frames in uint8 array form. 
        - output_path [str] - path to the folder you wish to save the video to
        - video_name [str, optional] - name of the croped video. By default name does not change
        - play_video [bool, optional] - option to play the resulting video.
    
    """
    
    if video_name is not None:
        output_path = output_path + "/" + video_name
    else:
        output_path = output_path + "/" + "new_video.mp4"
        
    f_shape = frames[0].shape
      
    
    # Get the frames per second (fps) and frame dimensions
    fps = fps
    
    # Check if cropping parameters are within bounds
    if width is None:
        test_width = 0
    if height is None:
        test_height = 0 
    else:
        test_height = height
        test_width = width
        
    if x < 0 or y < 0 or x + test_width > f_shape[1] or y + test_height > f_shape[0]: 
        print(f"Cropping parameters are out of bounds.\n Parameters: \n  x = {x} > 0 \n y = {y} > 0 \n x + width  = {x+test_width} < {f_shape[1]} \n y + height = {y+test_height} < {f_shape[0]}")
        return
    
    
    if width and height is not None:
        frame_width = width
        frame_height = height
    else:
        frame_width = f_shape[1]
        frame_height = f_shape[0]
    
    
    # Define the codec and create VideoWriter object to save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    #progress_bar initialization
    total_count = len(frames)
    print("Creating video. This will take a while...")
    progress_bar = tqdm(total = total_count, desc="Progress")

    
    for i in range(len(frames)):
        # Update the progress bar
        progress_bar.update()
        
        frame = frames[i]
        
        #Check for RGB
        if len(f_shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #Crop frame
        frame = frame[y:y+frame_height, x:x+frame_width]
        
        # Write the cropped frame to the output video
        out.write(frame)

    
    # Close the progress bar
    progress_bar.close()
    print(f"\nVideo {os.path.basename(output_path)} created sucesfully.")

    out.release()    
    
    if play_video:
        # Create a VideoCapture object for the video file
        cap = cv2.VideoCapture(output_path)
        
        # Check if the video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
        
        print("Video play starting. Press 'q' to exit the window.")
        # Get the original video dimensions (width, height)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the aspect ratio of the video
        aspect_ratio = original_width / original_height

        # Calculate the new width based on the desired screen height while maintaining the aspect ratio
        # Get the primary monitor's resolution
        monitor = get_monitors()[0]
        new_height = monitor.height
        new_width = int(aspect_ratio * new_height)
        
        
        while True:  # This creates an infinite loop to keep playing the video
            # Reset the capture object to start from the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
        
                # Break the inner loop if there are no more frames
                if not ret:
                    break
                # Resize the frame to fit within the desired screen size
                frame_resized = cv2.resize(frame, (new_width, new_height))
                
                # Display the frame
                cv2.imshow('Video Frame', frame_resized)
        
                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()  # Stop the program if 'q' is pressed
        
        # Release the VideoCapture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

def filter_names(names, exclude=None, include=None, filter_symbol=None):
    """
    Filters given list of names.
    
    Input parameters:
        - names [list] - List of names to filter.
        - exclude [list, optional] - List of names to exclude
        - include [list, optional] - List of names to include
        - filter_symbol [str, optionl] - Option to add a symbol that seperates different strings (eg. If you want to differentiate between 2 and 22, you can input a symbol after the end of the strng so that the new srings are 2[symbol] and 22[symbol])
    
    Output parameters:
        - filtered_names [list] - Filtered list of names. 
    """
    
    if exclude is None:
       exclude = []
    if include is None:
       include = []   
       

    
    filtered_names = []
    if filter_symbol is not None:
        exclude = [el + filter_symbol for el in exclude]
        include = [el + filter_symbol for el in include]
    for name in names:
      # Check if the name is not in the exclude list
        exclude_check = not any(re.search(sub, name, re.IGNORECASE) for sub in exclude)
        include_check = all(substring in name for substring in include)
        if exclude_check and include_check:
        # Check if the name contains all the substrings in include_name list
            filtered_names.append(name)

    return filtered_names

def compare_lists(list1, list2, mode="common"):
    """
    Compares two lists to see which elements are common or unique
    
    Input parameters:
        - list1 [list] - list you wish to compare
        - list2 [list] - list you wish to compare
        - mode  [string, optional] - mode by which it returns elements, default it is common. Modes are - common, unique, unique_first, unique_second.  
        Common mode returns elements that are present in both lists, unique_both mode returns elements that are different in both lists, 
        unique_first and unique_second returns elements that are present only in first or second list
    
    Output parameters:
        - lis3 [list] - list of values sorted by selected mode
    """
    if mode == 'common':
        return list(set(list1) & set(list2))
    elif mode == 'unique_both':
        return list(set(list1) ^ set(list2))
    elif mode == 'unique_first':
        return list(set(list1) - set(list2))
    elif mode == 'unique_second':
        return list(set(list2) - set(list1))
    else:
        raise ValueError("Invalid mode. Use 'common', 'unique_both', 'unique_first', or 'unique_second'.")
   
def getFiles_recursive(folder_path, suffix = None, include=None, exclude=None, search_recursive=True):
    """
    Recursively finds files with the specified suffix in the given directory and its subdirectories.

    Input parameters:
        - folder_path [string] - The path to the directory to start the search.
        - suffix [string, optional] - The suffix (file extension) to match. Default is None, suffix is then not checked
        - include [list, optional] - A list of names, that file names must include. If file name include one name but does not include a second it will be returned
        - exclude [list, optional] - A list of folder names or file names to exclude from the search.
        - search_recursive [bool, optional] - Option to change the searching pattern from recursive to only in current directory

    Output parameters:
        - list - A list of file names with the specified suffix found in the directory and its subdirectories.
    """
    if suffix is None:
        sf = ""
    else:
        sf = f".{suffix}"
    
    f_path = Path(folder_path)
    
    if search_recursive:
        files = list(f_path.rglob(f"*{sf}"))
    else:
        files = list(f_path.glob(f"*{sf}"))
        
    f_files =  []
    for i in files:
        if os.path.isfile(i):
            rel_path = Path(i).relative_to(Path(folder_path))
            f_files.append(str(rel_path))
        else:
            continue
    
    filt_rel = filter_names(f_files, exclude=exclude, include=include)   
    
    filtered_files = []
    for i in filt_rel:
       j = os.path.join(folder_path, i)
       filtered_files.append(j)
    
    return filtered_files

def get_folder_names(directory_path, exclude=None, include=None):
    """
    Retrieves the names of folders in the specified directory. It ignores everything that is not a folder

    Input parameters:
        - directory_path [string] - The path to the directory to retrieve folder names from.
        - exclude [list, optional] - A list of folder names to exclude from search
        - include [list, optional] - A list of folder names to include in search (folder name must contain this substring)

    Output parameters:
        - list - A list of folder names in the specified directory.
    """
    
    folder_names = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    filtered_fnames = filter_names(folder_names, exclude = exclude, include = include)
    return filtered_fnames

def getFolders_recursive(directory_path, include=None, exclude=None, search_recursive=True, end_folders_only=False):
    """
    Recursively finds files with the specified suffix in the given directory and its subdirectories.

    Input parameters:
        - directory_path [string] - The path to the directory to start the search.
        - include [list, optional] - A list of names, that file names must include. If file name include one name but does not include a second it will not be returned
        - exclude [list, optional] - A list of folder names or file names to exclude from the search.
        - recursive [bool, optional] - Option to change the searching pattern from recursive to only in current directory
        - end_folders_only [bool, optional] - Option to only get the deepest folders
    Output parameters:
        - list - A list of file names with the specified suffix found in the directory and its subdirectories.
    """
    
    
    if search_recursive:
        pattern = os.path.join(directory_path, "**/")
    else:
        pattern = os.path.join(directory_path, "*/")

    folder_paths = glob.glob(pattern, recursive=search_recursive)  

    folder_paths = [os.path.normpath(folder_path) for folder_path in folder_paths]
    
    if os.path.normpath(directory_path) in folder_paths:
        folder_paths.remove(os.path.normpath(directory_path))
    
    filtered_folder_paths = filter_names(folder_paths, exclude=exclude, include=include)
    
    final_folder_paths = filtered_folder_paths.copy()
    e = 0
    if end_folders_only:
        for i in filtered_folder_paths:
            if not get_folder_names(directory_path = i, exclude=exclude, include=include):
                e+=1
                continue
            else:
                del final_folder_paths[e]
    
    return final_folder_paths

def count_files_with_suffix(folder_path, suffix, exclude_folders = None):
    """
    Counts number of files in a folder with certain suffix
    
    Input parameters:
        - folder_path [string]- path to the folder, in which files are stored
        - suffix [string]- suffix of the files
    
    Output parameters:
        - number of files [float]
    """
    file_count = 0
    # Create the pattern to match files with the desired suffix
    pattern = os.path.join(folder_path, f"*{suffix}")
    # Use glob to find files matching the pattern
    matching_files = glob.glob(pattern)
    # Count the matching files
    file_count = len(matching_files)

    return file_count

def replace_substrings_in_file(file_path, replacements, print_report = False):
    """
    Replace specified substrings in a text file.
    
    Input parameters:
        - file_path - Path to the text file.
        - replacements - A list of tuples containing (old_substring, new_substring) pairs.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        for old_substring, new_substring in replacements:
            content = content.replace(old_substring, new_substring)

        with open(file_path, 'w') as file:
            file.write(content)
        if print_report:
            print("Replacements completed successfully.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def flatten_list(nested_list):
    """
    Gets rid of nested lists
    
    Input parameters:
        - nested_list [list]
    
    Output parameters:
        - flat list [list] - flattened list
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def flatten_and_remove_duplicates(list_of_lists):
    """
    Flatten a list of lists and remove duplicate elements, preserving their relative positions.

   Input paraleters:
       - list_of_lists [list] - Input list containing sublists.

   Output parameters:
       - list - Flattened list with unique elements in original order.
    """
    flattened_list = []
    seen_elements = set()

    for sublist in list_of_lists:
        for element in sublist:
            if element not in seen_elements:
                flattened_list.append(element)
                seen_elements.add(element)

    return flattened_list

def get_header_fromTXT(file_path, skip=0):
    """
    Returns header (first line) from a txt file.
    
    Input parameters:
        - file path [str] - path to the txt file
        - skip [int, optional] - option to skip x lines. Default 0.
    """
    
    cont = open(file = file_path, mode="r").readlines()
    cont = cont[skip::]
    head = cont[0].strip("\n")
    
    return head
    
def get_data_fromTXT(folder_path, skip=0, convert_to_float=False):
    """
    Returns list of data from a txt file.
    
    Input parameters:
        - folder_path [string] - path to the folder with txt files. If you wish to read only one txt file, specify the path to that file (it should include .txt suffix)
        - skip [int] - number of lines you wish to skip eg. txt. file has a header,... By default it is set to 0
        - convert_to_float [bool, optional] - True if you wish to convert values to floats.
    Output parameters:
        - data [dict, list] - dictionary of data or list of data in case of only one file. Keyword is file path, value is a list of data in a file
    """
    if ".txt" in folder_path:
        all_data =[]
        cont = open(file = folder_path, mode = "r").readlines()
        cont = cont[skip::]
        data = []
        for j in cont:
            if convert_to_float:
                number = float(j.strip("\n").replace(",", "."))
                data.append(number)
            else:
                data.append(j.rstrip("\n"))
        return data      
    else:
        files = glob.glob(f"{folder_path}/**.txt") 
        all_data = {}
        for i in files:
            cont = open(file = i, mode = "r").readlines()
            cont = cont[skip::]
            data = []
            for j in cont:
                if convert_to_float:
                    number = float(j.strip("\n").replace(",", "."))
                    print(number)
                    data.append(number)
                else:
                   data.append(j.rstrip("\n"))
            #saves data from one file to a dictionary. Key is name of the .txt file
            key = os.path.basename(i).strip(".txt")
            all_data.update({key : data})
        return all_data 
    
def convert_Qn_to_Q(Qn, rho_n, rho_real):
    """
    Converts standardised volume flow rate to real volume flow rate. 
    
    Input parameters:
        - Qn [float] - standardised volume flow rate
        - rho_n [float] - standard density. For air @ 0 °C and 1.01325 bar it is 1.293 kg/m^3
        - rho_real [float] - density of air at time of meassurement.
    
    Output parameters:
        - Q_real [float] - volume flow rate at density int the environment
        - M_meassured [float] - meassured mass flow rate 
    """
    
    mass_flow = Qn*rho_n
    volume_flow = mass_flow/rho_real
    
    return volume_flow, mass_flow
    
def get_pressure_data(folder_path, l_conv, g_conv, p_conv):
    """
    Imports and sorts pressure data in .txt files. 
    
    Input parameters:
        - folder_path - [string] path to the folder, where pressure data is located
        - l_conv - [float] factor with which data in file gets converted to standard units eg. 1 l/h ----> 2.77778 × 10^-7 m^3/s
        - g_conv - [float] factor with which data in file gets converted to standard units eg. 1 l/h ----> 2.77778 × 10^-7 m^3/s
        - p_conv- [float] factor with which data in file gets converted to standard units eg. 1 mbar ----> 1 x 10^2 Pa
        
    Output parameters:
        - averages - [dict] dictionary of data. Data is given as follows: key - name of the nozzle; data in order - liquid flow rate [int] (L, [m^3/s]), gas flow rate [int] (G, [m^3/s]), pressure [array] [Pa]
    """
    #Tukaj uvozim datoteke, shranim določene parametre in izračunam povprečja vrednosti tlaka v posameznih točkah
    averages = {}
    if os.path.isdir(folder_path):
        files = glob.glob(os.path.join(folder_path, "**.txt"), recursive=True)
    else:
        files = [folder_path]
    Filelenght=len(files)
    for i in range(0, Filelenght):
        RP = open(files[i], mode = "r").readlines()
        RP_l = []
        for k in RP:
            RP_l.append(float(k.rstrip("\n").replace("," , ".")) * p_conv)
        RP_ar = np.array((RP_l))
        GL= os.path.basename(files[i]).rstrip(".txt")
        L = int(GL.split("L")[-1]) * l_conv
        G = int(GL.split("-")[1].strip("G")) * g_conv
        _ = [G, L, RP_ar]
        _ = flatten_list(_)
        averages.update({GL : _})
    return averages


def get_pressure_data_labview(folder_path, p_conv, header_end="Untitled"):
    """
    Imports and sorts pressure data in .txt files. 
    
    Input parameters:
        - folder_path - [string] path to the folder, where pressure data is located
        - p_conv- [float] factor with which data in file gets converted to standard units eg. 1 mbar ----> 1 x 10^2 Pa
        - header_end - [string, optional] string by which the function knows where the header ends. It looks for a first instance of this string in the given txt files.

    Output parameters:
        pressure - [dict] Dictionary of pressure data

    """
    
    folders = getFiles_recursive(folder_path=folder_path, suffix="txt")
    P = {}

    for i in folders:
        
        nozzle_name = os.path.basename(i).strip(".txt")
        
        txt = open(i).readlines()
        h_e = next((j for j, s in enumerate(txt) if header_end in s), -1) +1

        txt = txt[h_e: ]
        p_values = np.array([float(s.split("\t")[1].strip("\n").replace(",", ".")) for s in txt])
        
        avg_p = np.mean(p_values) * p_conv

        P.update({nozzle_name: avg_p})
        
    return P
        
        
        

def import_pictures(pictures, resize = False, resize_factor=1):
    """
    Import pictures.
    
    Input parameters:
        - pictures [list] - List of picture locations.
    
    Output parameters:
        - picture_arrays [list(numpy_arrays)] - list of numpy arrays with RGB values in cv2 format.
    
    """
    print(1)    

    #progress_bar initialization
    total_count = len(pictures)
    print("Starting importing pictures. This will take a while...")
    progress_bar = tqdm(total = total_count, desc="Importing")

    frames = []
    for i in pictures:
        #progress bar update
        progress_bar.update()
        frame = Image.open(i)
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        if resize:
            img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
        frames.append(img)
    
    # Close the progress bar
    progress_bar.close()
    print("Pictures imported.")
    
    return frames


def extract_frames(video_path, output_folder=None, output_frames=False, frame_format="jpg", one_folder=False, error_path = None, report_existing_frames = False, limit_n_frames = None, grayscale=False):
    """
    Function to convert video file (in .mp4 format) to frames in specified format. 
    Requires OpenCV
    
    Input parameters:
        - video_path [string] - path to the folder or directly to the video, where video or videos are saved. If you wish to convert only one video you should specify the path only to that video
        - output_folder [string, optional] - path to the desired output folder. Folders with frames are saved in this folder. Folders are named after the videos. If left empty, frames will not be saved.
        - output_frames [bool, optional] - option to return a list of frames in array format.
        - frame_format [string] - format of the extracted frame. If no arguments are given, it defaults to .jpg format
        - one_folder [bool] - If True all frames will be saved in one folder, else it saves frames from each video in that videos folder. By default it is False.
        - error_path [str, optional] - path to the txt file, where error repots will be writen
        - report_existing_frames [bool, optional] - option to report if a frame with the same name as the one curently being saved already exists in the output folder.
        - limit_n_frames [int, optional] - Limits the number of frames the function extracts (eg. if you only need n number of frames)
        - grayscale [optional, bool] - converts RGB images to grayscale 
    
    Outpur parameters (optional):
        - frames [list] - list of frames in array format
    """
    if os.path.isdir(video_path):
        files = glob.glob(f"{video_path}/**.mp4")   
    else:
        files = [video_path]
    if files == []:
        a = video_path.split("/")
        a.insert(1, "\\")
        files = ["".join(a)]
    # Create the output folder if it doesn't exist
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    for i in files:
        # Open the video file
        video_capture = cv2.VideoCapture(i)
        # Check if video file opened successfully
        if not video_capture.isOpened():
            if error_path is not None:
                unsuccesfull = open(error_path, mode = "a")
                unsuccesfull.write(f"Video: {i} couldnt be opened. \n")
                unsuccesfull.close()
            print(f"Error opening video file: {i}")
            return 
        video_name = os.path.basename(i).rstrip(".mp4")
        # Initialize frame counter
        frame_count = 0
        # Read until the end of the video
        
        #progress_bar initialization
        total_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Starting frame extraction for {os.path.basename(i)}. This will take a while...")
        progress_bar = tqdm(total = total_count, desc="Extracting")
        
        if output_frames:
            output = []
        
        while video_capture.isOpened():
            #Update progress bar
            progress_bar.update()
            # Read a single frame from the video
            ret, frame = video_capture.read()
            # Break if no frame is retrieved
            if not ret:
                break
            #Save frames in output folder
            if output_frames:
                output.append(frame)
            if output_folder is not None:
                # Generate the output file path
                if one_folder == False:
                    file_path = os.path.join(f"{output_folder}/{video_name}", f"{video_name}_{frame_count:04d}.{frame_format}")
                    # Create the output folder named after the video
                    if not os.path.exists(f"{output_folder}/{video_name}"):
                        os.makedirs(f"{output_folder}/{video_name}")
                else:
                    file_path = os.path.join(f"{output_folder}", f"{video_name}_{frame_count:04d}.{frame_format}")
                    # Create the output folder named after the video
                    if not os.path.exists(f"{output_folder}"):
                        os.makedirs(f"{output_folder}")
                # Convert the frame to specified format and save it
                if not  os.path.exists(file_path): #check if frame already exists in output file
                    # Convert OpenCV BGR format to RGB format (imwrite ne dela, zato uporabljam PIL)
                    if grayscale:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert the frame to a PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    try:
                        pil_image.save(file_path)
                    except Exception as e:
                        (f"Failed to save the frame: {e}")
                        #cv2.imwrite(file_path, frame) does not work !!!!!!!!
                else:
                    if report_existing_frames:
                        print(f"Frame {video_name}_{frame_count:04d} already exists in output folder: {output_folder}")
            # Increment frame counter
            frame_count += 1
            if limit_n_frames is not None:
                if frame_count >= limit_n_frames:
                    break
                
        if output_frames:
            return output
        
        # Close the progress bar
        progress_bar.close()
        print(f"Extraction for {os.path.basename(i)} completed.")

        #Progress
        print(f"{files.index(i)+1}/{len(files)}")
        if len(files) == files.index(i)+1:
            print("---------------------END------------------------")    
        # Release the video file and close any open windows
        video_capture.release()
        cv2.destroyAllWindows()

def save_data(folder_path, data, names_of_data=None, file_name="data"):
    """
    Saves desired data to .txt file.
    
    Input parameters:
        - folder_path [string] - path to the folder in which data will be saved
        - data [list],[array],[dict] - array or list or dictionary of data to save
        - names_of_data [list] - names of groups of data. List will be inserted in the first row, names should be given in order of data they belong to, eg. data = [[1 ,2 ,3], [4, 5, 6]], names_of_data = ["123", "456"]
        - file_name [string, optional] - name of the .txt file. If left empty it will be named data.txt
    """
    f_name = f"{folder_path}/{file_name}.txt"
    ex = os.path.exists(f_name)
    if ex == True:
        file = open(f_name, mode = "a")
    else:
        file = open(f_name, mode = "w")
    if type(data) is not dict:
        if names_of_data is not None:
            file.write(str(names_of_data) + "\n")
        for i in data:
            file.write(str(i) + "\n")
        file.close()
    else:
        name=list(data)
        value=list(data.values())
        if names_of_data is not None:
            file.write(str(names_of_data + "\n"))
        for i in range(len(data)):
            file.write(str({name[i] : value[i]}) + "\n")
        file.close()
    return f"Data saved sucesfully. Path to the file: {folder_path}/{file_name}"

def rename_folders(directory, old_substring, new_substring, recursive = False, include=None, exclude = None, add_substring = None, report_missing = False, report_succes = False):
    """
    Renames folders in specified directory. Name changing can be recursive (optional), so folders that are saved 
    in folders inside the specified directory will be renamed. 
    
    It is recomended to make a backup copy before atempting to rename folders (just in case).
    
    Input parameters:
        
        - directory [string] - path to the folder
        - new_substring [string] - name or part of the name you wish to replace with
        - old_substring [string] - name or part of the name you wish to replace
        - recursive [bool, optional] - option to search folder names recursive
        - include [list, optional] - option to filter search by substring that must be present in the folder name
        - exclude [list, optional] - option to filter search by substring that must not be present in the folder name
        - add_substring [string. otional] - option to add a substring at the end of the folder name
        - report_missing [bool, optional] - option to report if old_substring is not found in folder name
        - report_succes [bool, optional] - option to report if filename is changed sucesfully
    """
    folders = getFolders_recursive(directory_path=directory, include=include, exclude=exclude, search_recursive=recursive)   
    
    for folder in folders:
        try:
            current_name = os.path.basename(folder)
            if old_substring not in current_name:
                if report_missing:
                    print(f"Substring {old_substring} not found in folder name :{folder}. Skipping.....")
                continue
            new_name = current_name.replace(old_substring, new_substring)
            if add_substring is not None:
                new_name = new_name + add_substring
            head = os.path.split(folder)[0]
            n_path = os.path.join(head, new_name) 
            os.rename(folder, n_path)
            if report_succes:
                print(f"Folder :{folder} renamed sucesfully.")
        except:
            raise Exception(f"Folder {folder} renamed unsucesfully")



def rename_files(directory, old_substring, new_substring, include=None, exclude=None, suffix = None, change_suffix = None, report_missing = False, report_succes = False):
    """
    Renames files in specified directory. Name changing is recursive, so files that are saved 
    in folders inside the specified directory will be renamed.
    
    It is recomended to make a backup copy before atempting to rename files (just in case).
    
    Input parameters:
        
        - directory [string] - path to the folder or file 
        - new_substring [string] - name or part of the name you wish to replace with
        - old_substring [string] - name or part of the name you wish to replace
        - include [list, optional] - option to filter search by substring that must be present in the file name
        - exclude [list, optional] - option to filter search by substring that must not be present in the file name
        - suffix [string, optional] - suffix of the files. Default is None.
        - change_suffix [string, optional] - Option to be able to change the suffix.
        - report_missing [bool, optional] - option to report if old_substring is not found in filename
        - report_succes [bool, optional] - option to report if filename is changed sucesfully
    """
    if os.path.isdir(directory):
        files = getFiles_recursive(folder_path=directory, suffix=suffix, include=include, exclude=exclude)   
    else:
        files = [directory]
    for file in files:
        try:
            current_name = os.path.basename(file).strip(f".{suffix}")
            if old_substring not in current_name:
                if report_missing:
                    print(f"Substring {old_substring} not found in filename :{file}. Skipping.....")
                continue
            if change_suffix is not None:
                int_name = current_name.replace(old_substring, new_substring)
                new_name = int_name + "." + change_suffix
            else:
                int_name = current_name.replace(old_substring, new_substring)
                new_name = int_name + "." + suffix
            head = os.path.split(file)[0]
            n_path = os.path.join(head, new_name) 
            os.rename(file, n_path)
            if report_succes:
                print(f"File :{file} renamed sucesfully.")
        except:
            raise Exception(f"File :{file} renamed unsucesfully")

def remove_duplicate_items_in_files(file_path):
    """
    Removes duplicated items in a specified file so that only one instance of item remains
    
    Input parameters:
        - file_path - path to the file
    """
    unique_strings = set()

    for file_path in glob.glob(file_path):
        filtered_lines = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    cleaned_line = line.strip()
                    if cleaned_line not in unique_strings:
                        unique_strings.add(cleaned_line)
                        filtered_lines.append(cleaned_line)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'w') as file:
            file.write('\n'.join(filtered_lines))
    return "Filtering completed successfully"

def strip_letters(input_string):
    """
    Removes all letter elements from input string
    
    Input parameters:
        - input_string [str] - String you wish to remove letters from
    
    Return parameters:
        - stripped_string [str]
    """
    # Create a translation table that removes letters
    translation_table = str.maketrans('', '', string.ascii_letters)
    # Use the translation table to remove letters from the input string
    stripped_string = input_string.translate(translation_table)
    return stripped_string

def sort_strings_by_key(strings):
    """
    Sort nozzle points by size. First it sorts by nozzle name, then value of G and lastly by value of L
    
    Input parameters:
        - strings [list] - list of strings (points) you wish to be sorted
        
    Output parameters:
        - sorted_strings [list]
    """
    def extract_key(item):
        point = item.split("-")
        first_value = float(strip_letters(point[0]))
        second_value = float(strip_letters(point[1]))
        third_value = float(strip_letters(point[-1]))
        return first_value, second_value, third_value
    
    return sorted(strings, key=extract_key)

def find_similar_values(data_dict, differentiation_threshold, differentiate_nozzles=False, nozzles_split=".", iterations_split="_"):
    """
    Finds values that are inside differentiation treshold and returns a dictionary of those values
    
    Input parameters:
        - data_dict [dict] - Dictionary of values. It should be writen as {base_name : value}
        - differentiation_treshold [float] - Treshold in which values can differentiate
        - differentiate_nozzles [bool, optional] - If True it writes only different nozzles (nozzles with same RP marking are not included when looking for differences)
        - nozzles_split [str, optional] - character that differs nozzles with the same geometrical values eg. RPx.1, RPx.2,...
        - iterations_split [str, optional] - character that differs iterations of data eg. RPx_1, RPx_2,... 
    
    Output parameters:
        - similar values [dict] - Dictionary in format; (key of main data : (main value : (key of compared value, compared value, difference))
    """
    
    similar_values = {}
    # Iterate through the data pairs
    for key1, value1 in data_dict.items():
        similar_keys = {}
        # Compare the current value with other values
        for key2, value2 in data_dict.items():
            if differentiate_nozzles is False:
                if key1 != key2 and abs(value1 - value2) <= differentiation_threshold:
                    similar_keys.update({value1: (key2, value2, value1-value2)})
            else:
                nozzle1 = key1.split(iterations_split)[0].split(nozzles_split)[0]
                nozzle2 = key2.split(iterations_split)[0].split(nozzles_split)[0]
                if nozzle1 != nozzle2 and abs(value1 - value2) <= differentiation_threshold:
                    similar_keys.update({value1: (key2, value2, value1-value2)})
        
        if similar_keys:
            similar_values[key1] = similar_keys
    
    if len(data_dict) == len(similar_values):
        print ("All data inside differentiation treshold")
    return similar_values

def find_values_in_dict(data_dict, mode, nozzles_split=".", iterations_split="_", **kwargs):
    """
    Compares values inside dictionary and depending on mode returns different dictionaries
    
    Input parameters:
        - data_dict [dict] - Dictionary of values. It should be writen as {base_name : value}
        - mode [str] - Mode by which function searches through dictionary. It can be; less, greater, equal_to or equal_in
        - nozzles_split [str, optional] - character that differs nozzles with the same geometrical values eg. RPx.1, RPx.2,...
        - iterations_split [str, optional] - character that differs iterations of data eg. RPx_1, RPx_2,... 
        - **kwargs:
            - If mode is less:
                - less_than [float] - uper limit of values
            - If mode is greater:
                - more_than [float] - lower limit of values
            - If mode is equal_to:
                - equal_to [float] - value that output values are equal to
            - If mode is equal_in:
                - base_value [float] - Value from which values in dict differentiate
                - differentiation_treshold [float] - Treshold in which values can differentiate

    Output parameters:
           - values [dict] - Dictionary in format; (key of value : value)
    """
    if mode == "less":        
        lesser_values = {}
        # Iterate through the data pairs
        for key, value in data_dict.items():
            if value < kwargs.get("less_than"):
                lesser_values.update({key : value})
        return lesser_values    
        
    if mode == "greater":
        greater_values = {}
        # Iterate through the data pairs
        for key, value in data_dict.items():
            if value > kwargs.get("more_than"):
                greater_values.update({key : value})
        return greater_values  
    
    if mode == "equal_to":
        equal_values = {}
        # Iterate through the data pairs
        for key, value in data_dict.items():
            if value == kwargs.get("equal_to"):
                equal_values.update({key : value})
        return equal_values  
    
    if mode == "equal_in":
        area = kwargs.get("differentiation_treshold")
        base_value = kwargs.get("base_value")
        equal_in = {}
        for key, value in data_dict.items():
            diff_value = value - base_value
            if abs(diff_value) < area:
                equal_in.update({key: (value, abs(diff_value))})
        return equal_in

def filter_dict_keys_by_substring(input_dict, substring, mode="include"):
    """
    Filters dictionary keys by specified substring. Returns dictionary containing filtered keys and their values
    
    Input parameters:
        - input_dict [dict] - dictionary through which the function iterates
        - substring [str] - substring that keys must contain
        - mode [str, optional] - mode by which function fiters keys. Modes are "include" and "exclude". In include mode keys must contain specified substring, in exclude mode they must not contain it.
        
    Output parameters:
        - filtered_dict [dict] - Dictionary containing only filtered keys and their coresponding values
    """
    filtered_dict = {}
    
    for key, value in input_dict.items():
        if mode == "include":
            if substring in key:
                filtered_dict[key] = value
        if mode == "exclude":
            if substring not in key:
                filtered_dict[key] = value
    return filtered_dict

def dictionaries_to_lists(*dictionaries):
    """
    Makes lists out of dictionaries with the same keys. Indexes of elements with matching keys are the same.
    
    Input_parameters:
        - dictionaries [dict] - any number of dictionaries. All dictionaries must have the same keys
    
    Output parameters:
        - lists [list] - Lists made out of input dictionaries. Lists are given in the same order that dictionaries were inputed.
    """
    # Check if dictionaries have the same keys
    keys = None
    for dictionary in dictionaries:
        if keys is None:
            keys = set(dictionary.keys())
        else:
            if set(dictionary.keys()) != keys:
                raise ValueError("Input dictionaries must have the same keys.")
    
    # Create lists for each dictionary
    lists = [[] for _ in range(len(dictionaries))]
    
    # Populate lists with values
    for key in keys:
        for i, dictionary in enumerate(dictionaries):
            lists[i].append(dictionary[key])
    
    return lists

def compare_dicts(dict1, dict2):
    """
    Compares dictionaries and returns dictionaries with only common keys (keys that are present in both original dictionaries), with their original values
    
    Input parameters:
        - dict1 [dict] - Dictionary 1 that you wish to compare
        - dict2 [dict] - Dictionary 2 that you wish to compare
    
    Output parameters:
        - common_dict_1 [dict] - dictionary with common keys and values coresponding to dict1 values of those keys
        - common_dict_2 [dict]- dictionary with common keys and values coresponding to dict2 values of those keys
    """
    # Find common keys in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Initialize dictionaries to store common key-value pairs from each dictionary
    common_dict1 = {}
    common_dict2 = {}

    # Iterate through common keys and populate the result dictionaries
    for key in common_keys:
        common_dict1[key] = dict1[key]
        common_dict2[key] = dict2[key]

    return common_dict1, common_dict2

def sort_dict_by_values(input_dict, reverse=False):
    """
    Sorts input dictionarie by values
    
    Input parameters:
        - input_dict [dict] - Dictionary you wish to sort
        - reverse [bool, optional] - reverse the order of sorting
    """
    # Sort the dictionary by values and return a list of key-value tuples
    sorted_items = sorted(input_dict.items(), key=lambda item: item[1], reverse=reverse)

    # Create a new dictionary from the sorted key-value tuples
    sorted_dict = dict(sorted_items)

    return sorted_dict

def sort_dict_values_by_elements(input_dict, reverse=False, sort_by_index=None):
    """
    Sorts dictionary where values are lists of numbers. It can sort by size of the numbers or by provided index changes list.
    
    Input parameters:
        - input_dict [dict] - Dictionary to be sorted
        - reverse [bool, optional] - Reverse the order of sorting by size.
        - sort_by_index [list, optional] - Nested list of index changes. If left empty, the function sorts by size. The number of nested lists must equal the number of keys with list values in the input dictionary.
   
   Output parameters:
       - sorted_dict [dict] - Sorted dictionary either by size or by index changes
       - index_changes [list, optional] - If sorted by size, a nested list of index changes
   """
    # Create a copy of the input dictionary
    sorted_dict = input_dict.copy()

    # Sort the values in lists based on element size (if sort_by_index is None)
    if sort_by_index is None:
        index_changes = []
        for key, value in sorted_dict.items():
            if isinstance(value, list):
                # Use sorted() to sort the list
                sorted_list = sorted(value, reverse=reverse)
                
                # Calculate index changes using enumerate()
                index_change_once = [sorted_list.index(value) for _, value in enumerate(value)]
                
                sorted_dict[key] = sorted_list
                index_changes.append(index_change_once)
                
        return sorted_dict, index_changes

    # Sort the values in lists based on a separate list of index changes (if provided)
    elif isinstance(sort_by_index, list):
        for key, index_changes in zip(sorted_dict.keys(), sort_by_index):
            if isinstance(sorted_dict[key], list):
                #list that coresponds to the current key in input dict
                original_list = sorted_dict[key]
                
                if len(original_list) != len(index_changes):
                    raise ValueError("Both lists must have the same length")
                # Create a list to store the reordered elements
                reordered_list = [None] * len(original_list)
                
                # Iterate through the index changes and update the reordered_list
                for i, change in enumerate(index_changes):
                    reordered_list[change] = original_list[i]
                sorted_dict[key] = reordered_list
        return sorted_dict

def find_keys_by_value(dictionary, value):
    """
    Find keys in a dictionary that have a specified value.

    Parameters:
        dictionary [dict] - input dictionary.
        value [/] - value to search for in the dictionary.

    Returns:
        keys [list] -  a list of keys in the dictionary that have the specified value.
    """
    return [key for key, val in dictionary.items() if val == value]

def dictionary_to_2d_array(input_dict):
    """
    Returns a 2D array where the first element is the key and the second is its coresponding value.
    
    Input parameters:
        - input_dict [dict] - Dictionary to transform
    
    Output parameters:
        - 2D array [list]
    """
    
    # Get the keys and values from the dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Create a 2D array by zipping the keys and values
    result = list(zip(keys, values))

    return result

def find_file_in_directory(file_name, directory, return_path = False):
    """
    Function to check if a file is in a specified directory.
    
    Input parameters:
        - file_name [str] - name of the file
        - directory [str] - directory to search inside
        - return_path [bool, optional] - option to return string containing file path
    
    Output parameters:
        - file_exists [bool] - True if file exists, False if not
        - file_path [str, optional] - Path to the file
    """
    file_path = None
    for file_path in glob.iglob(os.path.join(directory, '**', file_name), recursive=True):
        if return_path:
            return True, file_path
        else:    
            return True
    return False

def copy_file(source_file, destination_directory, report_state = True, return_path = False, unsuc_write=None):
    """
    Copy a file to a specified directory.

    Input parameters:
    - source_file [str] - Path to the source file.
    - destination_directory [str] - Path to the destination directory.
    - report_state [bool, optional] - Option to report the state of copying (copying succesfull or not).
    - return_path [bool, optional] - Option to return path to the copiesd file
    - unsuc_write [str, optional] - Option to write names of unsucesfull files in a txt file.

    Output parameters:
    - path_to_file [str, optional] - Path to the copied file in the destination directory.
    """
    try:
        file_name = os.path.basename(source_file)
        destination_file = os.path.join(destination_directory, file_name)
    
        shutil.copy(source_file, destination_file)
        
        if os.path.exists(destination_file):
            if report_state:
                print(f"File {file_name} copied sucesfully.")
            if return_path:
                return destination_file
        
        else:
            if report_state:
                print(f"Destination directory for {file_name} does not exist.")
            return None
        
    except Exception as e:
        print(f"Error copying file {file_name}: {e}")
        if unsuc_write is not None:
            unsuccesfull = open(unsuc_write, mode = "a")
            folder = os.path.split(source_file)[0]
            unsuccesfull.write(f"File {file_name} in folder {folder} couldnt be copied. \n")
            unsuccesfull.close()
        return None
    
def get_image_properties(image_path):
    """
    Returns properties of an image.

    Input parameters:
        - image_path[str] - path to the input image file.

    Output parameters:
        - image_parameters [dict] - dictionary containing properties of the image.
    """
    try:
        # Open the image file using PIL
        img = Image.open(image_path)

        # Extract properties
        properties = {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "dpi": img.info.get("dpi", None),
        }

        return properties
    except Exception as e:
        print(f"Error reading image properties: {e}")
        return None
    
def get_frame_properties(frame):
    """
    Returns properties of an image.

    Input parameters:
        - frame[list] - frame

    Output parameters:
        - image_parameters [dict] - dictionary containing properties of the image.
    """
    try:
        # Open the image file using PIL
        img = frame

        # Extract properties
        properties = {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "dpi": img.info.get("dpi", None),
        }

        return properties
    except Exception as e:
        print(f"Error reading image properties: {e}")
        return None

def get_video_properties(video_path):
    """
    Returns properties of a video file. 
    Properties are: frame_count, frame_width, frame_height, fps, duration_seconds (use these keys to search for them in return dict)

    Input parameters:
        - video_path [str] - path to the video file.

    Output parameters:
        - properties [dict] - dictionary containing video properties.
    """
    # Open video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return None

    # Get video properties
    properties = {
        'frame_count': int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        'frame_width': int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'frame_height': int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': video_capture.get(cv2.CAP_PROP_FPS),
        'duration_seconds': int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) / video_capture.get(cv2.CAP_PROP_FPS)
    }

    # Release video object
    video_capture.release()

    return properties



def adjust_brightness_frame(frame, factor):
    """
    Adjusts the brightness of an image.

    Input parameters:
        - frame[array] - frame in array format
        - factor [float] - brightness adjustment factor. Values greater than 1 increase brightness, while values between 0 and 1 decrease brightness.

    Output parameters:
        - adjusted frames [list, optional] - list of adjusted frames.
    """
    try:

        # Apply brightness adjustment
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

        return adjusted_frame
    except Exception as e:
        print(f"Error adjusting brightness: {e}")
        return None

def adjust_brightness_image(image_path, factor):
    """
    Adjusts the brightness of an image.

    Input parameters:
        - image_path [str] - path to the image.
        - factor [float] - brightness adjustment factor. Values greater than 1 increase brightness, while values between 0 and 1 decrease brightness.

    Output parameters:
        - adjusted frames [list, optional] - list of adjusted frames.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Apply brightness adjustment
        adjusted_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)

        return adjusted_img
    except Exception as e:
        print(f"Error adjusting brightness: {e}")
        return None

    

def adjust_brightness_video(video, alpha, beta, save_video = False, save_path = None):
    """
    Adjusts the brightness of a video.

    Input parameters:
        - video [str] -  path to the input video.
        - factor [float] - brightness adjustment factor. Values greater than 1 increase brightness, while values between 0 and 1 decrease brightness.
        - save_video [bool, optional] - option to save the video in mp4 format. If True, function will not return list of frames.
        - save_path [str, optional] - path to the adjusted video (only relevant if save_video is True)
        

    Output parameters:
        - adjusted frames [list, optional] - list of adjusted frames.
    """
    # Open video file
    video_capture = cv2.VideoCapture(video)
     
    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return []
     
    # Initialize list to store frames
    adjusted_frames = []
     
    # Process each frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
       
        #Apply brightnes adjustment to a frame
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        #Save adjusted frames in a list
        adjusted_frames.append(adjusted_frame)

    video_capture.release()
    
    if save_video:
        #getting video properties
        video_properties = get_video_properties(video_path=video)
        frame_width = video_properties.get("frame_width")
        frame_height = video_properties.get("frame_height")
        fps = video_properties.get("fps")
        
        #define save path for video
        if save_path is not None:
            root, ext = os.path.splitext(save_path)
            if ext.lower() == ".mp4":
                output_name = save_path
            else:
                output_name = os.path.join(root, ".mp4")
        else:
            input_folder = os.path.split(video)[0]
            output_name = os.path.join(input_folder, "output_video.mp4")
            
        #Define output video properties, writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_name, fourcc , fps, (frame_width, frame_height))
        
        for i in adjusted_frames:
            # Write each adjusted frame to output video
            output_video.write(i)
        
        #close the video
        output_video.release()
        
        return
    
    else:
        return adjusted_frames
    
    
def convert_one_frame_to_binary(frame_path):
    """
    Converts a frame to binary monochromatic image.
    
    Input parameters:
        - frame_path [str] - path to frame for conversion
        
    Output parameters:
        - converted_frame [array of uint8] - converted frame
    """
    try:
        # read the image file 
        img = cv2.imread(frame_path, 2)
          
        # converting to its binary form 
        ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    except img==None:
        print("Check input frame path.")
    
    return bw_img

def convertToBinary_otsu(frame):
    """
    Converts a frame to binary monochromatic image.
    
    Input parameters:
        - frame [str] - array of the frame you wish to binarise
        
    Output parameters:
        - converted_frame [array of uint8] - converted frame
    """
    assert frame is not None, "file could not be read, check with os.path.exists()"
    
    im1 = np.array(Image.open(frame).convert("L"))
    #Gaussian filtering blur
    blur = cv2.GaussianBlur(im1 ,(5, 5), 0)
    # converting to its binary form
    ret, bw_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return bw_img

def threshold_white(frame, filter_corner, filter_range):
    """
    Function to calculate the threshold to filter the image you wish o binarise. Useful with very thin jets.
    Input parameters:
        - frame [str or array] - path to the frame or the frame in array format
        - filter_corner [tuple] - position (x, y) of the upper left corner of the are in which the average white value will be calculated
        - filter-range [int] - length of the filter square size
    Output parameters:
        - threshold  [int] - threshold for the specified frame and range.
        - filter_image [array of uint8] - array of the filter range. Useful to check if it only includes background (as it should).
    """
    
    assert frame is not None, "file could not be read, check with os.path.exists()"
    
    if type(frame) == str:
        im1 = np.array(Image.open(frame).convert("L"))
    else:
        im1 = np.array(Image.fromarray(frame).convert("L"))
    
    filter_img = im1[filter_corner[1]:filter_corner[1]+filter_range:, 
                                      filter_corner[0]:filter_corner[0]+filter_range]
    if filter_img.size < 0:
        print("Invalid filter borders")
        
        
    white = np.round(np.average(filter_img))
    
    return white, filter_img
    
def convertToBinary_thin_jets(frame, filter_threshold=255, filter_offset=0, show_f_img = False):
    """
    Converts a frame to binary monochromatic image. Before binarisation a filter is applied, to better diferentiate 
    between white and black colors (minimization of background interference).
    
    Input parameters:
        - frame [str] - path to the image with the frame
        - filter_offset [int, optional] - value of the offset from the threshold.
        - filter_threshold [int, optional] - the threshold from which the program knows which pixels should be considered as white. Use function >>threshold_white<< to determine threshold.
        - show_f_img [bool, optional] - option to show the filtered image
    Output parameters:
        - converted_frame [array of uint8] - converted frame
    """
    assert frame is not None, "file could not be read, check with os.path.exists()"
    
    #im1 = np.array(Image.open(frame).convert("L"))[:,:7550]
    if type(frame) == str:
        im1 = np.array(Image.open(frame).convert("L"))
    else:
        im1 = np.array(Image.fromarray(frame).convert("L"))
        
    white = filter_threshold - filter_offset

    im2 = im1.copy()
    im2[im2 > white] = 255
    if show_f_img:
        Image.fromarray(im2).show()
    
    #Gaussian filtering blur
    #blur = im3.filter(ImageFilter.GaussianBlur(radius = 5)) 
    blur = cv2.GaussianBlur(im2 ,(5, 5), 0)
    # converting to its binary form
    ret, bw_img = cv2.threshold(np.array(blur), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return bw_img

    
def convert_multiple_frames_to_binary(directory, suffix, recursive_search=False, include=None, exclude=None):
    """
    Converts frames in a folder to binary monochromatic image.
    
    Input parameters:
        - directory [str] - path to the folder with frames for conversion
        - suffix [str] - suffix of frames
        - recursive search [bool, optional] - option to search recursively for frames in specified directory
        - include [list, optional] - convert only frames which names include a certain substring
        - exclude [list, optional] - convert only frames which name does not include a certain substring
    
    Output parameters:
        - converted_frames [list] - list with converted frames in binary form
    """
    
    all_frames_paths = getFiles_recursive(folder_path=directory, suffix=suffix, include=include, exclude=exclude, search_recursive=recursive_search)

    binary_frames = []
    for i in all_frames_paths:
        b_frame = convert_one_frame_to_binary(i)
        binary_frames.append(b_frame)
    
    return binary_frames

def binary_to_jpg(binary_data, output_path):
    """
    Converts binary image data to JPEG format and saves it to a file.

    Input parameters:
        - binary_string [list] -  Binary image data (zeros and ones not bytes)
        - output_path [str] -  Path to save the output JPEG file.
    """
    try:
        # Create a BytesIO object from binary data
        stream = io.BytesIO(binary_data)

        # Open binary data as an image using PIL
        image = Image.open(stream)

        # Save the image as JPEG
        image.save(output_path, format='JPEG')
        print("Image successfully converted to JPEG.")
    except Exception as e:
        print(f"Error converting binary data to JPEG: {e}")
        
def array_to_image(array, location=None, return_image = False):
    """
    Saves a picture in np.array form into a RGB image object.
    
    Input parameters:
        - array [numpy_array] - array with RGB picture values. It must have uint8 values.
        - location [str, optional] - location to save the resulting image.
        - return_image [bool, optional] - option to get an Image objet as a return argument.
    """
    
    im = Image.fromarray(array).convert("RGB")
    if location is not None:
        im.save(location)
    if return_image:
        return im
    
  
def clean_image_bcg(image, sqr_size=10, threshold=100):
    """
    Cleans the background of an image. 
    Background is determined by the average value of a square of pixels.
    
    Input parameters:
        - image [np.array] - array of an image in RGB format.
        - sqr_size [int, optional] - number of px that a square size has.
        - threshold [float, optional] - threshold to determine if a square is part of the bcg or not.
    
    Output paramters:
        - sub_image [np.array] - array of the cleaned image.
    """
    
    n = image.copy()
    m = n.copy()
    
    
    kvadr = sqr_size
    
    #progress_bar initialization
    total_count = len(np.concatenate(n))
    progress_bar = tqdm(total = total_count, desc="Cleaning the image")
    
    
    for i in range(len(m)):
        for j in range(len(m[i])):
            progress_bar.update()
            
            kv = m[i-kvadr-1:i+kvadr, j-kvadr-1:j+kvadr]
            kv_average = np.average(kv)
            
            if kv_average > threshold:
                n[i][j] = np.array([255, 255, 255])
    
    progress_bar.close()
    
    return n
    


def find_nozzle_center(image, nzl_end, nzl_width, threshold=20, sqr_size=10, crop_image=False, show_center=False):
    """
    Finds the centre of the nozzle.
    
    Input parameters:
        - image [np.array] - array of an image (should be in RGB format).
        - nzl_end [int] - px value whre the nozzle ends. Use "defineStartJetExp" function.
        - nzl_width [int] - px value of half of the width of the nozzle
        - threshold [float, optional] - threshold by which the function determines which px belong to the nozzle. Lower if background is very dark.
        - sqr_size [int, optional] - size of a square that is used in calculation.
        - crop_image [bool] - option to get a croped image as an output
        - show_center [bool] - option to get an image with the nzl. center marked as an output.
        
    Output parameters:
        - nzl_center [int] - px value of the nzl. center. If crop_image is True, this value takes into acount the croping.
        - croped_image [np.array, optional] - croped image (left and right by the value of the nzl. width)
        - marked_image [np.array, optional] - image with the nzl. center marked.
    """
    m = image.copy()
    
    upper_mod = image.copy()[:nzl_end]
    
    crop_upper = upper_mod.copy()
    
    kvadr_nzl = sqr_size
    
    #progress_bar initialization
    total_count = nzl_end*len(upper_mod[0]) + nzl_end*len(m[0])
    progress_bar = tqdm(total = total_count, desc="Finding nzl. center")
    
    for i in range(0, nzl_end):
        for j in range(len(upper_mod[i])):
            progress_bar.update()
            
            kv = m[i-kvadr_nzl-1:i+kvadr_nzl, j-kvadr_nzl-1:j+kvadr_nzl]
            kv_average = np.average(kv)
            
            if kv_average < threshold:
                crop_upper[i][j] = np.array([255, 0, 0])
            else:
                crop_upper[i][j] = np.array([0, 0, 0])
    
    kvadr_pos = int(nzl_end/2)
    nozzle_center = 0
    kv_avg_prev = 0
    for i in range(0, nzl_end):
        for j in range(len(m[i])):
            
            progress_bar.update()
            kv_avg = np.average(crop_upper[i-kvadr_pos-1: i+kvadr_pos, j-kvadr_pos-1:j+kvadr_pos])
            
            if kv_avg > kv_avg_prev:
                nozzle_center = j
            
            kv_avg_prev = kv_avg
            
    output = []
    output.append(nozzle_center)
    
    if crop_image:
        croped_bin_image = m[:, nozzle_center-nzl_width-1 : nozzle_center+nzl_width]
        
        output.append(croped_bin_image)
    
    if show_center:
        marked_img = image.copy()
        marked_img[:, nozzle_center] = np.array([255, 0, 0])
        
        output.append(marked_img)
        
    progress_bar.close()

    return output


def send_email(subject, body, to_email):
    # Email credentials
    from_email = "jernej.kusar6@gmail.com"
    password = "nrdz hoxx hlar gjlp"  # Use an app password, not your actual password

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Send the email via Gmail SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
  