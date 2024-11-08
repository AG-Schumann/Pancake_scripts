import cv2
import numpy as np
import argparse
import os
import subprocess


def compute_frame_brightness(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    frame_avg_brightness = []
    frame_max_brightness = []

    brightest_frame = None
    mb = -1
    frame_index = 0
    brightest_frame_index = -1
    
    # Check if the video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        # Read one frame at a time
        ret, frame = video_capture.read()
        
        # If the frame is not read successfully, end of video is reached
        if not ret:
            break
        
        # Convert the frame to grayscale to measure brightness
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the average brightness (mean of the pixel values)
        avg_brightness = np.mean(gray_frame)

        # max brightness
        max_brightness = np.amax(gray_frame)

      # Check if this is the brightest frame so far
        if max_brightness > mb:
          mb = max_brightness
          brightest_frame = frame.copy()  # Store the brightest frame
          brightest_frame_index = frame_index

        frame_index += 1
        
        # Store the brightness value
        frame_avg_brightness.append(avg_brightness)
        frame_max_brightness.append(max_brightness)

    # Release the video capture object
    video_capture.release()

       
    return np.array(frame_avg_brightness), np.array(frame_max_brightness), brightest_frame



def look_at_frames(frame_numbers, video_path, opath, display_results=False):
  # Open the video file
  # video_path = "WIN_20241023_16_41_29_Pro.mp4"
  cap = cv2.VideoCapture(video_path)

  for frame_number in frame_numbers:

    # Specify the frame number you want to jump to
    #frame_number = 2477
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the specified frame
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale to find the brightest pixel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the brightest pixel in the frame
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)


        # Display the result
        if display_results:
          cv2.imshow("Brightest Pixel Indicated", frame)
          cv2.waitKey(0)
          cv2.destroyAllWindows()

        # Optionally, save the image
        cv2.imwrite(f"{opath}/{frame_number}.png", frame)

        # Draw a circle around the brightest pixel
        cv2.circle(frame, max_loc, radius=20, color=(0, 0, 255), thickness=2)
        cv2.imwrite(f"{opath}/{frame_number}_circ.png", frame)

    else:
        print(f"Frame {frame_number} could not be read.")

  # Release the video capture object
  cap.release()


if __name__ == '__main__':
  parser=argparse.ArgumentParser(
    description='''Average / Max brighntess per frame.''',
    epilog="""Let's see....""")
  #parser.add_argument('-f','--file', type=argparse.FileType('r'), default=None, required=True, action='store', help='video file (.mp4)')
  parser.add_argument('filename', help='Input video file (.mp4)')
  #parser.add_argument('-o','--out', type=str, default=None, help='filename of plot output')
  parser.add_argument('-o','--out', type=str, default=None, help='path for output files')
  #parser.add_argument('-e','--extract_frames', action='store_true', help='Scan entire')
  #parser.add_argument('-c','--draw_circle', action='store_true', help='Draw circle around brightest pixel')
  parser.add_argument(
    "-f",
    "--frames",
    type=int,
    default=[],
    nargs="*",  # Accepts 0 or more integers
    help="Extract only the frames of the given ids"
  )
  parser.add_argument('-t','--threshold', type=float, default=0.1, help='Threshold to select bright frames')
  parser.add_argument('-v','--verbose', action='store_true', help='Enable verbose output')
  args=parser.parse_args()

  if args.verbose:
    print (args)
  video_path = args.filename
  if args.filename and len(args.frames)==0:
    f_avg, f_max, bf = compute_frame_brightness(video_path)

    if args.out:
      #ofn=args.out
      if os.path.isdir(args.out):
        opath=args.out
    else:
      #ofn=f"{args.filename}_plotoutput"
      opath = os.path.dirname( video_path )

    if bf is not None:
      output_file = f'brightest_frame.png'
      cv2.imwrite(os.path.join(opath,output_file), bf)

    #if args.plot:
    if True: 
      import matplotlib.pyplot as plt
      fid = np.arange(len(f_avg))
      plt.plot(fid, f_avg/np.amax(f_avg), 'o-', color='b', label='avg')
      plt.plot(fid, f_max/np.amax(f_max), 'o-', color='r', label='max')
      ## print index of all bright frames:
      bright,  = np.where ( f_max/np.amax(f_max) > args.threshold )
      plt.text(0.5, 0.5, f"bright frames: {bright}")
      #plt.show()
      plt.xlabel('Frame No')
      plt.legend()
      plt.tight_layout()
      #plt.savefig(opath+'brightness_vs_frame.png')
      plt.savefig(os.path.join(opath,'brightness_vs_frame.png'))

    # now also extract the brightest images (if less than 10 for now)
    if len(bright) < 11 and True:
      look_at_frames(frame_numbers=bright, video_path=args.filename, opath=opath, display_results=False)

      
  elif not args.filename:
    if args.verbose:
      print (f"No video file specified. Break.")
    parser.print_help()
  else:
    if args.out:
      if os.path.isdir(args.out):
        opath=args.out
    else:
      opath = os.path.dirname( args.filename )
    look_at_frames(frame_numbers = args.frames, video_path=args.filename, opath=opath, display_results=False)
