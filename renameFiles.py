#!/usr/bin/python

import os
import sys

# Function to rename multiple files
def main():
   i = 1
   path = input("Enter path to bulk rename folder: ")
   
   #path="E:/Outside Fiverr/uziel/Object Identifying Project/Training/cascade_training_part2_try1/n/"
   print("\nThis will rename all files at "+path)
   print("Do you wish to continute? (Y/N)", end="")
   consent = input()
   if consent!='Y':
      sys.exit()
   for filename in os.listdir(path):
      my_dest =str(i) + ".jpg"
      my_source =path + filename
      my_dest =path + my_dest
      # rename() function will
      # rename all the files
      try:
         os.rename(my_source, my_dest)
      except Exception:
         continue
      i += 1
   print(str(i)+" files renamed.")
   print("Press enter to exit.")
   exiting = input()
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()